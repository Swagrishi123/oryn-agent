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
    "answer directly and concisely in text. Do NOT run valuation models or tools.\n"
    "2. For valuation and DCF requests, you MUST execute python_interpreter. "
    "Never estimate numbers or write valuation formulas in text without running code.\n"
    "3. Output clean Markdown tables and executive summaries.\n"
    "   - NEVER use backticks (`) around currency, numbers, or phrases.\n"
    "   - Format currency values as '$346M' or '346 million USD'."
)

VALUATION_KEYWORDS = [
    "dcf", "valuation", "wacc", "discount rate", "ebitda",
    "terminal growth", "sensitivity matrix", "financial model",
    "cash flow", "excel", "lbo", "irr", "npv", "unlevered"
]

CACHED_WORKING_MODEL = None

def extract_code_from_markdown(text: str) -> str | None:
    m = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m_generic = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if m_generic:
        return m_generic.group(1).strip()
    return None

def sanitize_report_formatting(text: str) -> str:
    cleaned = re.sub(r'```(?:python)?\s*[\s\S]*?```', '', text)
    cleaned = cleaned.replace("** `", "**").replace("` **", "**")
    cleaned = cleaned.replace("`**", "**").replace("**`", "**")
    cleaned = re.sub(r'`([^`\n]+)`', r'\1', cleaned)
    cleaned = cleaned.replace('`', '')
    return re.sub(r'\n{3,}', '\n\n', cleaned).strip()

def enforce_sandbox_contracts(code_str: str) -> str:
    # Ensure Excel output always points to model.xlsx
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
    global CACHED_WORKING_MODEL
    if CACHED_WORKING_MODEL:
        return CACHED_WORKING_MODEL

    try:
        all_ids = [m.id for m in client.models.list().data]
    except Exception:
        all_ids = []

    valid_candidates = [
        mid for mid in all_ids
        if "/" not in mid and not any(b in mid.lower() for b in ["guard", "whisper", "vision", "embed", "audio", "orpheus"])
    ]

    priority = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama3-70b-8192",
        "llama-3.1-8b-instant",
        "llama-3.2-3b-preview",
        "deepseek-r1-distill-llama-70b",
        "mixtral-8x7b-32768"
    ]

    ordered = [p for p in priority if p in valid_candidates] + [c for c in valid_candidates if c not in priority]

    for cand in ordered:
        try:
            client.chat.completions.create(
                model=cand,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
            CACHED_WORKING_MODEL = cand
            return cand
        except Exception:
            continue

    CACHED_WORKING_MODEL = "llama-3.1-8b-instant"
    return CACHED_WORKING_MODEL

def execute_with_self_healing(client: Groq, model: str, initial_code: str, hitl_enabled: bool, status_callback=None):
    safe_code = enforce_sandbox_contracts(initial_code)
    out, p_bytes, x_bytes = tool_python_interpreter(safe_code, hitl_enabled=hitl_enabled)

    if "Python Execution Error" in out or "Error:" in out:
        if status_callback:
            status_callback("info", "Self-healing: fixing script syntax in sandbox...")
        repair_prompt = (
            f"The following Python script threw an error:\n{out}\n\n"
            f"Script:\n```python\n{safe_code}\n```\n\n"
            "Fix the syntax error. Make sure all parentheses and brackets match. "
            "Ensure it saves a matplotlib plot and exports 'model.xlsx'. "
            "Return ONLY the fixed code inside ```python ... ```."
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": repair_prompt}],
                max_tokens=2048
            )
            repaired = extract_code_from_markdown(resp.choices[0].message.content or "")
            if repaired:
                safe_repaired = enforce_sandbox_contracts(repaired)
                out, p_bytes, x_bytes = tool_python_interpreter(safe_repaired, hitl_enabled=hitl_enabled)
        except Exception:
            pass

    return out, p_bytes, x_bytes

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
        status_callback("info", f"Connected to Groq Engine: `{active_model}`")

    is_valuation_query = any(k in user_input.lower() for k in VALUATION_KEYWORDS)

    # 1. Non-quantitative requests (e.g. "capital of China")
    if not is_valuation_query:
        try:
            resp = client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=1000
            )
            elapsed = time.perf_counter() - start_time
            return sanitize_report_formatting(resp.choices[0].message.content or ""), None, None, elapsed
        except Exception as e:
            return f"Error: {e}", None, None, time.perf_counter() - start_time

    # 2. Quantitative valuation pipeline
    generated_plot = None
    generated_excel = None
    tool_outputs = []

    if status_callback:
        status_callback("info", "Executing quantitative calculations & rendering charts...")

    tool_executed = False

    # Attempt 1: Native Groq tool calling with tool_choice="required"
    try:
        response = client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            tools=AGENT_TOOLS,
            tool_choice="required",
            max_tokens=2048
        )
        msg = response.choices[0].message
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc.function.name == "python_interpreter":
                    try:
                        args = json.loads(tc.function.arguments)
                        raw_code = args.get("code", "")
                    except Exception:
                        raw_code = ""
                    if raw_code:
                        out, p_bytes, x_bytes = execute_with_self_healing(client, active_model, raw_code, hitl_enabled, status_callback)
                        if p_bytes: generated_plot = p_bytes
                        if x_bytes: generated_excel = x_bytes
                        tool_outputs.append(f"Calculation Data:\n{out}")
                        tool_executed = True
    except Exception:
        tool_executed = False

    # Attempt 2: Strict isolated Python generator fallback
    if not tool_executed or not tool_outputs:
        if status_callback:
            status_callback("info", "Synthesizing quantitative Python model...")

        pure_code_prompt = (
            f"Write a standalone, self-contained Python script to calculate this financial model:\n"
            f"{user_input}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Define parameters: base_revenue, growth_rate, ebitda_margin, wacc, terminal_growth, years = 5.\n"
            "2. Project Revenues, EBITDAs, and discount them by (1 + wacc)**t to get PV of UFCF.\n"
            "3. Calculate Terminal Value using Gordon Growth and discount it to present value.\n"
            "4. Calculate Implied Enterprise Value = sum(PV of UFCF) + PV of Terminal Value.\n"
            "5. Build a sensitivity matrix varying WACC and Terminal Growth.\n"
            "6. Generate a matplotlib bar chart (plt.figure(); plt.bar(['PV of Cash Flows', 'PV of Terminal Value'], [pv_sum, pv_tv]); plt.title('Valuation Breakdown ($M)'); plt.tight_layout()).\n"
            "7. Save the model and sensitivity matrix to 'model.xlsx' using pd.ExcelWriter('model.xlsx', engine='openpyxl').\n"
            "8. Print the projection table and summary values.\n"
            "OUTPUT STRICTLY ONLY PYTHON CODE INSIDE ```python ... ```."
        )

        try:
            resp = client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": "You are a Python code generator. Output ONLY Python code inside ```python ``` blocks. Do not write explanations."},
                    {"role": "user", "content": pure_code_prompt}
                ],
                max_tokens=2048,
                temperature=0.1
            )
            code = extract_code_from_markdown(resp.choices[0].message.content or "")
            if code:
                out, p_bytes, x_bytes = execute_with_self_healing(client, active_model, code, hitl_enabled, status_callback)
                if p_bytes: generated_plot = p_bytes
                if x_bytes: generated_excel = x_bytes
                tool_outputs.append(f"Calculation Data:\n{out}")
        except Exception as e:
            tool_outputs.append(f"Execution note: {e}")

    # 3. Final Executive Briefing Synthesis
    if not tool_outputs:
        elapsed = time.perf_counter() - start_time
        return "⚠️ **Execution Error**: Could not execute Python calculations. Please try again.", None, None, elapsed

    if status_callback:
        status_callback("info", "Formulating executive valuation briefing...")

    combined_results = "\n\n".join(tool_outputs)

    try:
        synth = client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"User request: {user_input}\n\n"
                        f"Deterministic sandbox calculations:\n{combined_results}\n\n"
                        "BRIEFING REQUIREMENTS:\n"
                        "- Provide an executive valuation summary.\n"
                        "- Include the 5-Year Projection Table, Terminal Value breakdown, and Sensitivity Matrix in clean Markdown tables.\n"
                        "- Use ONLY the calculated numbers from the sandbox.\n"
                        "- DO NOT output Python code, scripts, or imports.\n"
                        "- DO NOT wrap numbers or sentences in backticks (`)."
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
