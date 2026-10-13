import json
import os
import re
import time
from groq import Groq
from tools import AGENT_TOOLS, tool_get_financial_data, tool_web_search
from sandbox import tool_python_interpreter

SYSTEM_PROMPT = (
    "You are ORYN Analyst, an institutional BFSI quantitative analyst.\n"
    "CORE RULES:\n"
    "1. For valuation and DCF models, compute deterministic cash flows, discounting, and sensitivity matrices.\n"
    "2. CRITICAL PYTHON RULE: All code MUST be 100% self-contained.\n"
    "   - Explicitly define ALL parameters at the top (base_revenue, growth_rate, ebitda_margin, wacc, terminal_growth, years = 5).\n"
    "   - Always import pandas as pd, numpy as np, and matplotlib.pyplot as plt.\n"
    "3. Output clean Markdown tables and executive summaries."
)

def clean_extracted_code(raw_code: str) -> str:
    clean = raw_code.replace('\\\\n', '\n').replace('\\n', '\n')
    clean = clean.replace(r'\"', '"').replace(r"\'", "'").replace(r'\\', '\\')
    lines = clean.split('\n')
    valid = [l for l in lines if l.strip() not in ['PY', 'PY"', '}', '}}', '```', '```python']]
    return '\n'.join(valid).strip()

def extract_code_from_error(err_str: str) -> str | None:
    # Only recover if failed_generation is explicitly present in the error string
    if "failed_generation" not in err_str:
        return None
    m_gen = re.search(r"['\"]failed_generation['\"]\s*:\s*['\"](.*?)['\"]", err_str, re.DOTALL)
    if not m_gen:
        return None
    search_target = m_gen.group(1)
    m_code = re.search(r'["\']code["\']\s*:\s*["\'](.*?)(?:["\']\s*\}|\Z)', search_target, re.DOTALL)
    if m_code:
        code_text = clean_extracted_code(m_code.group(1))
        # Ensure it actually looks like Python code, not an error word
        if any(kw in code_text for kw in ['import', '=', 'def', 'plt', 'pd', 'np']):
            return code_text
    return None

def extract_code_from_markdown(text: str) -> str | None:
    """Extracts python code blocks from model responses."""
    m = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m_generic = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if m_generic:
        return m_generic.group(1).strip()
    return None

def wrap_with_safe_preamble(code_str: str) -> str:
    preamble = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "model_terms_required = 5\n"
        "years = 5\n"
        "projection_years = 5\n"
        "forecast_period = 5\n"
    )
    return preamble + "\n" + code_str

def resolve_api_key(api_key: str = "") -> str:
    if api_key and api_key.strip():
        return api_key.strip()
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()

def get_best_available_model(client: Groq) -> str:
    """Finds the best active chat/text model on this specific Groq key."""
    try:
        all_models = [m.id for m in client.models.list().data]
    except Exception:
        all_models = []

    # Filter out audio, vision, guard moderation models
    valid_models = [
        m for m in all_models 
        if not any(b in m.lower() for b in ["guard", "whisper", "vision", "embed", "audio"])
    ]

    # Preference ranking for Groq models
    priority = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-3b-preview",
        "llama-3.2-1b-preview",
        "qwen-2.5-32b",
        "deepseek-r1-distill-llama-70b",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]

    for cand in priority:
        if cand in valid_models:
            return cand

    return valid_models[0] if valid_models else "llama-3.1-8b-instant"

def run_analyst_agent(
    user_input: str,
    api_key: str = "",
    selected_model: str = None,
    engine_mode: str = "Cloud (Groq BYOK)",
    hitl_enabled: bool = True,
    status_callback=None,
    **kwargs
) -> tuple[str, bytes | None, bytes | None, float]:
    start_time = time.perf_counter()

    active_key = resolve_api_key(api_key)
    if not active_key:
        return "⚠️ **API Key Missing**: Please provide a Groq API key.", None, None, 0.0

    try:
        client = Groq(api_key=active_key, max_retries=0)
    except Exception as e:
        return f"⚠️ **Client Error**: {e}", None, None, 0.0

    active_model = get_best_available_model(client)
    if status_callback:
        status_callback("info", f"Connected to Groq Engine: `{active_model}`")

    generated_plot = None
    generated_excel = None
    tool_outputs = []
    final_text = ""

    # Attempt 1: Standard Tool Calling Execution
    tool_calling_succeeded = False
    try:
        response = client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            tools=AGENT_TOOLS,
            tool_choice="auto",
            max_tokens=1500
        )
        msg = response.choices[0].message
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                if fn_name == "python_interpreter":
                    if status_callback:
                        status_callback("info", "Executing Python quantitative sandbox...")
                    code_to_run = wrap_with_safe_preamble(args.get("code", ""))
                    out, p_bytes, x_bytes = tool_python_interpreter(code_to_run, hitl_enabled=hitl_enabled)
                    if p_bytes: generated_plot = p_bytes
                    if x_bytes: generated_excel = x_bytes
                    tool_outputs.append(f"[python_interpreter]:\n{out}")
            tool_calling_succeeded = True
    except Exception as e:
        # Check if error contains tool failure
        err_msg = str(e)
        recovered = extract_code_from_error(err_msg)
        if recovered:
            if status_callback:
                status_callback("info", "Executing recovered quantitative code...")
            safe_code = wrap_with_safe_preamble(recovered)
            out, p_bytes, x_bytes = tool_python_interpreter(safe_code, hitl_enabled=hitl_enabled)
            if p_bytes: generated_plot = p_bytes
            if x_bytes: generated_excel = x_bytes
            tool_outputs.append(f"[python_interpreter]:\n{out}")
            tool_calling_succeeded = True

    # Attempt 2: Universal Fallback (if model does not support tool calling)
    if not tool_calling_succeeded or not tool_outputs:
        if status_callback:
            status_callback("info", "Synthesizing quantitative DCF calculations...")

        code_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"User request: {user_input}\n\n"
            "Generate complete Python code to compute this valuation model, save sensitivity plots via matplotlib, "
            "and create 'model.xlsx' with openpyxl/pandas. "
            "Wrap your code strictly inside ```python ... ``` fences."
        )

        resp = client.chat.completions.create(
            model=active_model,
            messages=[{"role": "user", "content": code_prompt}],
            max_tokens=1500
        )
        content = resp.choices[0].message.content or ""
        code = extract_code_from_markdown(content)
        if code:
            if status_callback:
                status_callback("info", "Running quantitative sandbox...")
            safe_code = wrap_with_safe_preamble(code)
            out, p_bytes, x_bytes = tool_python_interpreter(safe_code, hitl_enabled=hitl_enabled)
            if p_bytes: generated_plot = p_bytes
            if x_bytes: generated_excel = x_bytes
            tool_outputs.append(f"[python_interpreter]:\n{out}")

    # Final Briefing Generation
    if status_callback:
        status_callback("info", "Generating final executive valuation briefing...")

    combined_results = "\n\n".join(tool_outputs) if tool_outputs else "Valuation calculations complete."
    try:
        synth = client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"User prompt: {user_input}\n\nCalculation Results:\n{combined_results}\n\nFormat these results into a clean executive Markdown valuation summary with valuation tables."}
            ],
            max_tokens=1500
        )
        final_text = synth.choices[0].message.content or combined_results
    except Exception:
        final_text = f"### 📊 Valuation Analysis Completed\n\n{combined_results}"

    elapsed = time.perf_counter() - start_time
    return final_text, generated_plot, generated_excel, elapsed
