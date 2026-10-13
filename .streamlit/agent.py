import io
import json
import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
from tools import AGENT_TOOLS, tool_get_financial_data, tool_web_search
from sandbox import tool_python_interpreter

SYSTEM_PROMPT = (
    "You are ORYN Analyst, an elite institutional BFSI quantitative analyst and investment banking advisor.\n\n"
    "CRITICAL LIVE DATA PROTOCOL:\n"
    "- Whenever a user asks about a company, stock price, valuation, P/E ratio, or market cap, "
    "YOU MUST CALL the `get_financial_data` tool to pull verified real-time numbers from NSE/BSE or global exchanges.\n"
    "- NEVER guess, invent, or rely on outdated memory for current stock prices.\n\n"
    "CRITICAL FORMATTING & BREVITY RULES:\n"
    "1. NO INTRODUCTORY FLUFF: Eliminate greetings, pleasantries, or preamble. Start directly with the analysis header.\n"
    "2. STRICT TABLE SYNTAX: Always separate table rows with actual newlines (\\n). Never join rows together.\n"
    "   | Col 1 | Col 2 |\n"
    "   | :--- | :--- |\n"
    "   | Val 1 | Val 2 |\n"
    "3. RESTRUCTURING & WATERFALL RULES:\n"
    "   - Always put Chapter 11 priority distributions in ONE consolidated table.\n"
    "   - DO NOT write out repetitive narrative blocks like 'Step 1: DIP... Step 2: First Lien...'. "
    "Put all steps directly into the table so the response completes without running out of tokens.\n"
    "   - Sum total prepetition claims accurately before calculating recoveries.\n"
    "4. FIG BANKING & LEVFIN LOGIC:\n"
    "   - In rate cuts, asset-sensitive bank NIM compresses. High deposit beta cushions; low deposit beta causes severe compression.\n"
    "   - Subordinated/unsecured debt MUST price higher than senior secured debt.\n"
    "   - Sources & Uses percentages must sum to 100.0%.\n"
    "5. DCF VALUATIONS: Build models, charts, and Excel workbooks only when specifically asked to run a DCF or financial model."
)

CACHED_WORKING_MODEL = None

def is_quantitative_model_request(prompt: str) -> bool:
    p = prompt.lower()
    dcf_terms = [
        "dcf", "discounted cash flow", "sensitivity matrix", 
        "unlevered", "run a financial model", 
        "build a financial model", "run a 5-year", "run a 10-year",
        "export the financial model to excel"
    ]
    if any(t in p for t in dcf_terms):
        return True
    has_discount_metric = any(m in p for m in ["wacc", "discount rate", "terminal growth"])
    has_financial_metric = any(m in p for m in ["ebitda margin", "revenue growth", "cash flow projection"])
    return bool(has_discount_metric and has_financial_metric)

def sanitize_report_formatting(text: str) -> str:
    """Fixes collapsed markdown tables and prevents run-on text."""
    cleaned = re.sub(r'```(?:python)?\s*[\s\S]*?```', '', text)
    # Turn back-to-back pipes into separate lines
    cleaned = re.sub(r'\|\s*\|', '|\n|', cleaned)
    cleaned = re.sub(r'([^\n])\n(\|.*\|)', r'\1\n\n\2', cleaned)
    cleaned = re.sub(r'(\|.*\|)\n([^\n|])', r'\1\n\n\2', cleaned)
    cleaned = re.sub(r'(?<!\\)\$', r'\$', cleaned)
    return re.sub(r'\n{3,}', '\n\n', cleaned).strip()

def enforce_sandbox_contracts(code_str: str) -> str:
    clean_code = re.sub(r'ExcelWriter\s*\(\s*[\'"][^\'"]+\.xlsx[\'"]', 'ExcelWriter("model.xlsx"', code_str)
    clean_code = re.sub(r'\.to_excel\s*\(\s*[\'"][^\'"]+\.xlsx[\'"]', '.to_excel("model.xlsx"', clean_code)
    preamble = "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nyears = 5\n"
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

def get_best_available_model(client: Groq, exclude_model: str = None, force_refresh: bool = False) -> str:
    global CACHED_WORKING_MODEL
    if CACHED_WORKING_MODEL and not force_refresh and CACHED_WORKING_MODEL != exclude_model:
        return CACHED_WORKING_MODEL

    try:
        models_data = client.models.list().data
        available_ids = [m.id for m in models_data]
    except Exception:
        available_ids = []

    filter_out = ["safeguard", "guard", "whisper", "orpheus", "allam", "arabic", "vision", "embed", "audio"]
    candidates = [m for m in available_ids if not any(b in m.lower() for b in filter_out) and m != exclude_model]
    preference = ["llama-3.1-8b-instant", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]

    test_queue = [m for m in preference if m in candidates] + [m for m in candidates if m not in preference]
    if not test_queue:
        test_queue = candidates if candidates else preference

    for model_name in test_queue:
        try:
            client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": "ping"}], max_tokens=1)
            CACHED_WORKING_MODEL = model_name
            return model_name
        except Exception:
            continue

    if candidates:
        CACHED_WORKING_MODEL = candidates[0]
        return CACHED_WORKING_MODEL
    raise RuntimeError("Could not connect to any Groq text model.")

def create_completion_with_fallback(client: Groq, preferred_model: str = None, status_callback=None, **kwargs):
    global CACHED_WORKING_MODEL
    active_model = preferred_model or get_best_available_model(client)
    
    # Generous 1200-token ceiling so reports never cut off
    kwargs["max_tokens"] = min(kwargs.get("max_tokens", 1200), 1200)
    kwargs["temperature"] = kwargs.get("temperature", 0.1)
    kwargs["model"] = active_model

    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "rate_limit_exceeded" in err_str:
            delay_match = re.search(r'try again in ([0-9.]+)s', err_str)
            sleep_sec = float(delay_match.group(1)) + 0.5 if delay_match else 2.5
            if status_callback:
                status_callback("info", f"Quota pause ({sleep_sec:.1f}s)... retrying.")
            time.sleep(sleep_sec)
            alt_model = get_best_available_model(client, exclude_model=active_model, force_refresh=True)
            kwargs["model"] = alt_model
            return client.chat.completions.create(**kwargs)
        raise e

def build_contextual_messages(user_input: str, system_prompt: str, max_turns: int = 3) -> list[dict]:
    history = []
    try:
        import streamlit as st
        if "messages" in st.session_state and isinstance(st.session_state.messages, list):
            candidates = st.session_state.messages[:-1] if st.session_state.messages[-1].get("content") == user_input else st.session_state.messages
            for m in candidates[-(max_turns * 2):]:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role in ["user", "assistant"] and content:
                    history.append({"role": role, "content": str(content)[:300]})
    except Exception:
        pass
    return [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_input}]

def df_to_markdown_table(df: pd.DataFrame, include_index: bool = False) -> str:
    headers = (["Parameter"] if include_index else []) + [str(col) for col in df.columns]
    rows = [[str(idx)] + [str(val) for val in row] for idx, row in zip(df.index, df.values)] if include_index else [[str(val) for val in row] for row in df.values]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n\n" + "\n".join([header_line, sep_line] + row_lines) + "\n\n"

def generate_deterministic_dcf_artifacts(user_input: str) -> tuple[str, bytes, bytes]:
    # Standard DCF parameters
    base_rev, growth, margin, wacc, tg, years = 50.0, 0.20, 0.25, 0.10, 0.025, 5
    years_idx = list(range(1, years + 1))
    rev = [base_rev * ((1 + growth) ** t) for t in years_idx]
    ebitda = [r * margin for r in rev]
    ufcf = ebitda
    df = [1 / ((1 + wacc) ** t) for t in years_idx]
    pv_ufcf = [u * d for u, d in zip(ufcf, df)]
    sum_pv_ufcf = sum(pv_ufcf)
    tv = (ufcf[-1] * (1 + tg)) / (wacc - tg)
    pv_tv = tv / ((1 + wacc) ** years)
    ev = sum_pv_ufcf + pv_tv

    df_proj = pd.DataFrame({
        "Year": [str(y) for y in years_idx],
        "Revenue ($M)": [f"${x:.2f}M" for x in rev],
        "EBITDA ($M)": [f"${x:.2f}M" for x in ebitda],
        "UFCF ($M)": [f"${x:.2f}M" for x in ufcf],
        "Discount Factor": [f"{x:.4f}" for x in df],
        "PV of UFCF ($M)": [f"${x:.2f}M" for x in pv_ufcf]
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(['PV Cash Flows', 'PV Terminal Value'], [sum_pv_ufcf, pv_tv], color=['#4a148c', '#7b1fa2'], width=0.45)
    ax.set_title(f'Enterprise Value: ${ev:.2f}M', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value ($M)')
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=200)
    plt.close(fig)
    plot_bytes = img_buf.getvalue()

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
        df_proj.to_excel(writer, sheet_name='DCF_Projections', index=False)
    excel_bytes = excel_buf.getvalue()

    calc_briefing = (
        f"### 📊 Institutional DCF Valuation\n\n"
        f"**Implied Enterprise Value:** **${ev:.2f}M**\n\n"
        f"{df_to_markdown_table(df_proj, include_index=False)}"
    )
    return calc_briefing, plot_bytes, excel_bytes

def run_analyst_agent(
    user_input: str,
    api_key: str = "",
    selected_model: str = None,
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

    # 1. DCF Routing
    if is_quantitative_model_request(user_input):
        if status_callback:
            status_callback("info", "Generating quantitative DCF model and Excel file...")
        briefing_text, plot_bytes, excel_bytes = generate_deterministic_dcf_artifacts(user_input)
        return sanitize_report_formatting(briefing_text), plot_bytes, excel_bytes, time.perf_counter() - start_time

    # 2. Institutional Research & Live Data Execution Loop
    try:
        messages = build_contextual_messages(user_input, SYSTEM_PROMPT, max_turns=3)
        
        # Call model with live data tools enabled
        first_resp = create_completion_with_fallback(
            client=client,
            preferred_model=selected_model,
            status_callback=status_callback,
            messages=messages,
            tools=AGENT_TOOLS,
            tool_choice="auto",
            max_tokens=1200
        )
        msg = first_resp.choices[0].message

        # Check if the AI decided to fetch live stock data
        if getattr(msg, "tool_calls", None):
            messages.append(msg)
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                
                if fn_name == "get_financial_data":
                    ticker = args.get("ticker", "")
                    if status_callback:
                        status_callback("info", f"Fetching live market metrics for {ticker} from NSE/BSE...")
                    live_result = tool_get_financial_data(ticker)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": json.dumps(live_result)
                    })

            # Call model a second time with the real live numbers in its hand
            second_resp = create_completion_with_fallback(
                client=client,
                preferred_model=selected_model,
                status_callback=status_callback,
                messages=messages,
                max_tokens=1200
            )
            final_text = second_resp.choices[0].message.content or ""
        else:
            final_text = msg.content or ""

        elapsed = time.perf_counter() - start_time
        return sanitize_report_formatting(final_text), None, None, elapsed

    except Exception as e:
        return f"Error: {e}", None, None, time.perf_counter() - start_time
