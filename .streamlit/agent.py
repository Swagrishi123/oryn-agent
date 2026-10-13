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
    "1. For general or conversational questions (e.g. 'What is the capital of China?'), "
    "answer directly and concisely. Do NOT invoke valuation models or tools.\n"
    "2. For valuation and DCF requests, ALWAYS execute python_interpreter to calculate cash flows, "
    "create a matplotlib chart, and write 'model.xlsx'.\n"
    "3. CRITICAL PYTHON RULE: All code in python_interpreter MUST be 100% self-contained.\n"
    "   - Explicitly define ALL parameters at the top (base_revenue, growth_rate, ebitda_margin, wacc, terminal_growth, years = 5).\n"
    "   - Always import pandas as pd, numpy as np, and matplotlib.pyplot as plt.\n"
    "   - ALWAYS create a visual plot using plt.figure() and plt.bar() or plt.plot().\n"
    "   - ALWAYS save the excel file strictly as 'model.xlsx' using pd.ExcelWriter('model.xlsx', engine='openpyxl').\n"
    "4. Output clean Markdown tables and executive summaries.\n"
    "   - NEVER use backticks (`) around currency, numbers, or sentences.\n"
    "   - Format currency values cleanly as '$346M' or '346 million USD'."
)

VALUATION_KEYWORDS = [
    "dcf", "valuation", "wacc", "discount rate", "ebitda",
    "terminal growth", "sensitivity matrix", "financial model",
    "cash flow", "excel", "lbo", "irr", "npv", "unlevered"
]

def clean_extracted_code(raw_code: str) -> str:
    clean = raw_code.replace('\\\\n', '\n').replace('\\n', '\n')
    clean = clean.replace(r'\"', '"').replace(r"\'", "'").replace(r'\\', '\\')
    lines = clean.split('\n')
    valid = [l for l in lines if l.strip() not in ['PY', 'PY"', '}', '}}', '```', '```python']]
    return '\n'.join(valid).strip()

def extract_code_from_error(err_str: str) -> str | None:
    if "failed_generation" not in err_str:
        return None
    m_gen = re.search(r"['\"]failed_generation['\"]\s*:\s*['\"](.*?)['\"]", err_str, re.DOTALL)
    if not m_gen:
        return None
    search_target = m_gen.group(1)
    m_code = re.search(r'["\']code["\']\s*:\s*["\'](.*?)(?:["\']\s*\}|\Z)', search_target, re.DOTALL)
    if m_code:
        code_text = clean_extracted_code(m_code.group(1))
        if any(kw in code_text for kw in ['import', '=', 'def', 'plt', 'pd', 'np']):
            return code_text
    return None

def extract_code_from_markdown(text: str) -> str | None:
    m = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m_generic = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if m_generic:
        return m_generic.group(1).strip()
    return None

def sanitize_report_formatting(text: str) -> str:
    """Removes leaked Python blocks and cleans stray backticks causing green highlight bugs."""
    # 1. Remove raw Python code blocks
    cleaned = re.sub(r'```(?:python)?\s*[\s\S]*?```', '', text)
    
    # 2. Fix broken formatting like `346 million** or **Value of `
    cleaned = cleaned.replace("** `", "**").replace("` **", "**")
    cleaned = cleaned.replace("`**", "**").replace("**`", "**")
    
    # 3. Strip standalone backticks that wrap numbers/words and create green pills
    cleaned = re.sub(r'`([^`\n]+)`', r'\1', cleaned)
    cleaned = cleaned.replace('`', '')
    
    return re.sub(r'\n{3,}', '\n\n', cleaned).strip()

def enforce_sandbox_contracts(code_str: str) -> str:
    """Enforces standard sandbox contracts: guarantees 'model.xlsx' and a matplotlib chart."""
    clean_code = re.sub(r'ExcelWriter\s*\(\s*[\'"][^\'"]+\.xlsx[\'"]', 'ExcelWriter("model.xlsx"', code_str)
    clean_code = re.sub(r'\.to_excel\s*\(\s*[\'"][^\'"]+\.xlsx[\'"]', '.to_excel("model.xlsx"', clean_code)

    preamble = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "model_terms_required = 5\n"
        "years = 5\n"
        "projection_years = 5\n"
        "forecast_period = 5\n"
    )
    return preamble + "\n" + clean_code

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
    try:
        all_models = [m.id for m in client.models.list().data]
    except Exception:
        all_models = []

    valid_models = [
        m for m in all_models 
        if not any(b in m.lower() for b in ["guard", "whisper", "vision", "embed", "audio"])
    ]

    priority = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-3b-preview",
        "deepseek-r1-distill-llama-70b",
        "mixtral-8x7b-32768"
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
        return "⚠️ **API Key Missing**: Please provide a Groq API key in the sidebar.", None, None, 0.0

    try:
        client = Groq(api_key=active_key, max_retries=0)
    except Exception as e:
        return f"⚠️ **Client Error**: {e}", None, None, 0.0

    active_model = get_best_available_model(client)
    if status_callback:
        status_callback("info", f"Executing on engine: `{active_model}`")

    is_valuation_query = any(k in user_input.lower() for k in VALUATION_KEYWORDS)

    generated_plot = None
    generated_excel = None
    tool_outputs = []

    # --- STEP 1: Tool Execution ---
    # For quantitative valuation requests, require python_interpreter so the LLM cannot skip it.
    forced_choice = {"type": "function", "function": {"name": "python_interpreter"}} if is_valuation_query else "auto"

    tool_calling_succeeded = False
    try:
        response = client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            tools=AGENT_TOOLS,
            tool_choice=forced_choice,
            max_tokens=2048
        )
        msg = response.choices[0].message

        # Direct response for non-valuation queries (e.g. "capital of China")
        if not getattr(msg, "tool_calls", None) and not is_valuation_query:
            elapsed = time.perf_counter() - start_time
            return sanitize_report_formatting(msg.content or ""), None, None, elapsed

        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                if fn_name == "python_interpreter":
                    if status_callback:
                        status_callback("info", "Executing quantitative calculations & rendering charts...")
                    safe_code = enforce_sandbox_contracts(args.get("code", ""))
                    out, p_bytes, x_bytes = tool_python_interpreter(safe_code, hitl_enabled=hitl_enabled)
                    if p_bytes: generated_plot = p_bytes
                    if x_bytes: generated_excel = x_bytes
                    tool_outputs.append(f"Calculation Data:\n{out}")
                elif fn_name == "get_financial_data":
                    res = tool_get_financial_data(args.get("ticker", ""))
                    tool_outputs.append(f"Financial Data:\n{res}")
                elif fn_name == "web_search":
                    res = tool_web_search(args.get("query", ""))
                    tool_outputs.append(f"Search Results:\n{res}")
            tool_calling_succeeded = True

    except Exception as e:
        recovered = extract_code_from_error(str(e))
        if recovered:
            if status_callback:
                status_callback("info", "Executing quantitative model...")
            safe_code = enforce_sandbox_contracts(recovered)
            out, p_bytes, x_bytes = tool_python_interpreter(safe_code, hitl_enabled=hitl_enabled)
            if p_bytes: generated_plot = p_bytes
            if x_bytes: generated_excel = x_bytes
            tool_outputs.append(f"Calculation Data:\n{out}")
            tool_calling_succeeded = True

    # --- STEP 2: Fallback (if tool calling was not triggered) ---
    if is_valuation_query and not tool_outputs:
        if status_callback:
            status_callback("info", "Generating quantitative valuation model...")

        code_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"User request: {user_input}\n\n"
            "REQUIREMENTS:\n"
            "1. Write complete Python code to compute the DCF valuation.\n"
            "2. Generate a clean visual plot with matplotlib (e.g. plt.figure(); plt.bar(...); plt.title(...)).\n"
            "3. Export the model and sensitivity tables to Excel strictly as 'model.xlsx' using pd.ExcelWriter('model.xlsx', engine='openpyxl').\n"
            "Wrap all code strictly inside ```python ... ``` fences."
        )

        try:
            resp = client.chat.completions.create(
                model=active_model,
                messages=[{"role": "user", "content": code_prompt}],
                max_tokens=2048
            )
            code = extract_code_from_markdown(resp.choices[0].message.content or "")
            if code:
                if status_callback:
                    status_callback("info", "Executing Python sandbox...")
                safe_code = enforce_sandbox_contracts(code)
                out, p_bytes, x_bytes = tool_python_interpreter(safe_code, hitl_enabled=hitl_enabled)
                if p_bytes: generated_plot = p_bytes
                if x_bytes: generated_excel = x_bytes
                tool_outputs.append(f"Calculation Data:\n{out}")
        except Exception as e:
            tool_outputs.append(f"Execution note: {e}")

    # --- STEP 3: Final Executive Synthesis ---
    if status_callback:
        status_callback("info", "Formulating executive briefing...")

    combined_results = "\n\n".join(tool_outputs) if tool_outputs else ""

    if not is_valuation_query and not combined_results:
        elapsed = time.perf_counter() - start_time
        return "I am ORYN Analyst. How can I assist with your quantitative modeling today?", None, None, elapsed

    try:
        synth = client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"User request: {user_input}\n\n"
                        f"Sandbox Calculations:\n{combined_results}\n\n"
                        "FORMAT INSTRUCTIONS:\n"
                        "- Provide an executive valuation summary and complete 5-Year Projection Table.\n"
                        "- Include the Terminal Value breakdown and Sensitivity Matrix.\n"
                        "- STRICT RULE: Do not output Python code blocks or imports.\n"
                        "- STRICT RULE: Do not wrap numbers or phrases in backticks (`). Use standard text formatting."
                    )
                }
            ],
            max_tokens=2500
        )
        raw_text = synth.choices[0].message.content or combined_results
        final_text = sanitize_report_formatting(raw_text)
    except Exception as e:
        final_text = sanitize_report_formatting(f"### 📊 Valuation Analysis Completed\n\n{combined_results}")

    elapsed = time.perf_counter() - start_time
    return final_text, generated_plot, generated_excel, elapsed
