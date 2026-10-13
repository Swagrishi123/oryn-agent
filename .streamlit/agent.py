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
    "1. For valuation and DCF models, compute deterministic cash flows, discounting, and sensitivity matrices using python_interpreter.\n"
    "2. CRITICAL PYTHON RULE: All code executed in python_interpreter MUST be 100% self-contained and standalone.\n"
    "   - Explicitly define ALL variables at the top of the script (e.g. base_revenue = 50e6, growth_rate = 0.20, ebitda_margin = 0.25, wacc = 0.10, terminal_growth = 0.025, years = 5).\n"
    "   - NEVER reference undefined variables.\n"
    "   - Always import pandas as pd, numpy as np, and matplotlib.pyplot as plt.\n"
    "3. Round all percentages to 2 decimal places and currency values appropriately.\n"
    "4. Output clean Markdown tables and executive summaries. Do not include raw code scripts, LaTeX blocks ($$), or HTML tags in the final briefing."
)

def clean_extracted_code(raw_code: str) -> str:
    clean = raw_code.replace('\\\\n', '\n').replace('\\n', '\n')
    clean = clean.replace(r'\"', '"').replace(r"\'", "'").replace(r'\\', '\\')
    lines = clean.split('\n')
    valid = [l for l in lines if l.strip() not in ['PY', 'PY"', '}', '}}', '```', '```python']]
    return '\n'.join(valid).strip()

def extract_code_from_error(err_str: str) -> str | None:
    m_gen = re.search(r"['\"]failed_generation['\"]\s*:\s*['\"](.*?)['\"]\s*\}", err_str, re.DOTALL)
    search_target = m_gen.group(1) if m_gen else err_str
    m_code = re.search(r'["\']code["\']\s*:\s*["\'](.*?)(?:["\']\s*\}|\Z)', search_target, re.DOTALL)
    return clean_extracted_code(m_code.group(1)) if m_code else None

def wrap_with_safe_preamble(code_str: str) -> str:
    """Guarantees common quantitative modeling variables exist so the interpreter never hits NameError."""
    preamble = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "model_terms_required = 5\n"
        "years = 5\n"
        "projection_years = 5\n"
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

def get_best_available_model(client: Groq, requested_model: str = None) -> str:
    chat_models = []
    try:
        for m in client.models.list().data:
            mid = m.id.lower()
            if any(blocked in mid for blocked in ["guard", "whisper", "vision", "embed", "audio"]):
                continue
            chat_models.append(m.id)
    except Exception:
        pass

    if requested_model and requested_model in chat_models:
        return requested_model

    priority_order = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-3b-preview",
        "deepseek-r1-distill-llama-70b",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    for candidate in priority_order:
        if candidate in chat_models:
            return candidate

    return chat_models[0] if chat_models else "llama-3.1-8b-instant"

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
        return (
            "⚠️ **API Key Missing**: Please enter your Groq API key in the sidebar.",
            None,
            None,
            0.0
        )

    try:
        client = Groq(api_key=active_key, max_retries=0)
    except Exception as e:
        return f"⚠️ **Client Initialization Error**: {e}", None, None, 0.0

    active_model = get_best_available_model(client, selected_model)
    if status_callback:
        status_callback("info", f"Executing on engine: `{active_model}`")

    scratchpad = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    generated_plot = None
    generated_excel = None
    tool_outputs = []
    final_text = ""

    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=active_model,
                messages=scratchpad,
                tools=AGENT_TOOLS,
                tool_choice="auto",
                max_tokens=1500
            )
        except Exception as e:
            err = str(e)
            recovered = extract_code_from_error(err)
            if recovered:
                if status_callback:
                    status_callback("info", "Executing recovered quantitative code...")
                safe_code = wrap_with_safe_preamble(recovered)
                out, p_bytes, x_bytes = tool_python_interpreter(safe_code, hitl_enabled=hitl_enabled)
                if p_bytes: generated_plot = p_bytes
                if x_bytes: generated_excel = x_bytes
                tool_outputs.append(f"[python_interpreter]:\n{out}")
                scratchpad.append({"role": "tool", "tool_call_id": "call_rec", "name": "python_interpreter", "content": out})
                break
            return f"Execution error: {err}", None, None, time.perf_counter() - start_time

        msg = response.choices[0].message
        scratchpad.append(msg)

        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}

                if status_callback:
                    status_callback("info", f"Executing `{fn_name}`...")

                if fn_name == "python_interpreter":
                    code_to_run = wrap_with_safe_preamble(args.get("code", ""))
                    out, p_bytes, x_bytes = tool_python_interpreter(code_to_run, hitl_enabled=hitl_enabled)
                    if p_bytes: generated_plot = p_bytes
                    if x_bytes: generated_excel = x_bytes
                    res = out
                elif fn_name == "get_financial_data":
                    res = tool_get_financial_data(args.get("ticker", ""))
                elif fn_name == "web_search":
                    res = tool_web_search(args.get("query", ""))
                else:
                    res = f"Unknown tool: {fn_name}"

                tool_outputs.append(f"[{fn_name}]:\n{res}")
                scratchpad.append({"role": "tool", "tool_call_id": tc.id, "name": fn_name, "content": res})
        else:
            final_text = msg.content or ""
            break

    # Final Executive Synthesis
    if tool_outputs and not final_text:
        if status_callback:
            status_callback("info", "Synthesizing executive valuation briefing...")
        try:
            synth = client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Query: {user_input}\n\nTool Results:\n" + "\n\n".join(tool_outputs) + "\n\nProvide the final executive valuation briefing in clean Markdown tables."}
                ],
                max_tokens=1500
            )
            final_text = synth.choices[0].message.content or "Analysis complete."
        except Exception:
            formatted_outputs = "\n\n".join(tool_outputs)
            final_text = f"### 📊 Valuation Analysis Completed\n\n{formatted_outputs}"

    elapsed = time.perf_counter() - start_time
    return final_text, generated_plot, generated_excel, elapsed
