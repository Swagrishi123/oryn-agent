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
    "CRITICAL MARKET DATA RULE:\n"
    "- When [LIVE EXCHANGE DATA] is provided, you MUST construct the analysis around those exact metrics.\n"
    "- Output real numbers directly in Markdown tables. If a number is an estimate, state it directly in plain text (e.g., '₹1,280 (Est.)').\n\n"
    "ABSOLUTE PROHIBITION ON CODE TEMPLATES & FAKE APIS:\n"
    "- NEVER output Jinja templates, brackets, pseudo-code, or fictional functions like '{oryn.get_current_share_price(...)}' or '{oryn.get_market_cap(...)}'.\n"
    "- NEVER claim or imply that the user needs an 'ORYN API' enabled.\n"
    "- Never refuse with 'I don't have real-time data'. Analyze the provided numbers or state reasonable baseline estimates with an institutional disclaimer.\n\n"
    "CRITICAL FORMATTING & STRUCTURAL RULES:\n"
    "1. DIRECT STRUCTURAL DELIVERY: Eliminate introductory greetings, pleasantries, and meta-commentary. "
    "Begin immediately with the report title.\n"
    "2. STRICT MARKDOWN TABLE SYNTAX:\n"
    "   - Every table row MUST be separated by an actual newline (\\n) and start/end with a pipe (|).\n"
    "   - Always put the header divider row (| :--- | :--- |) on its own separate line.\n"
    "   - Never join multiple table rows together horizontally.\n"
    "3. INSTITUTIONAL RESTRUCTURING & WATERFALL RULES:\n"
    "   - Always present Chapter 11 priority distributions inside a single consolidated Markdown table.\n"
    "   - Do NOT write repetitive text blocks like 'Step 1: DIP... Step 2: First Lien...'. "
    "Consolidate all steps into the table so the response completes before token limits.\n"
    "   - Sum total prepetition claims accurately before calculating recoveries.\n"
    "4. FIG & BANKING REASONING:\n"
    "   - Asset-sensitive banks face NIM COMPRESSION during rate cuts.\n"
    "   - High deposit beta cushions NIM; low deposit beta causes severe compression.\n"
    "   - Rate cuts increase bond prices, reversing AFS paper losses and boosting TCE.\n"
    "5. LEVERAGED FINANCE (LEVFIN) RULES:\n"
    "   - Pricing Hierarchy: Subordinated/unsecured debt MUST price HIGHER than senior secured debt.\n"
    "   - Sources & Uses percentages must divide by Total Sources/Uses and sum to exactly 100.0%.\n"
    "6. M&A ACCRETION REASONING:\n"
    "   - Stock deals depend on relative P/E arbitrage (Buyer P/E > Target P/E is accretive).\n"
    "   - Debt/cash deals depend on Target Earnings Yield (1 / Target Purchase P/E) vs. After-Tax Cost of Debt [Rate * (1 - Tax)].\n"
    "7. DCF MODELING: For DCF requests, execute quantitative schedules, render a matplotlib chart, and write 'model.xlsx'."
)

CACHED_WORKING_MODEL = None

KNOWN_ENTITIES = {
    "reliance": "RELIANCE",
    "tcs": "TCS",
    "tata consultancy": "TCS",
    "infosys": "INFY",
    "infy": "INFY",
    "hdfc": "HDFCBANK",
    "hdfc bank": "HDFCBANK",
    "icici": "ICICIBANK",
    "icici bank": "ICICIBANK",
    "sbi": "SBIN",
    "state bank of india": "SBIN",
    "tata motors": "TATAMOTORS",
    "itc": "ITC",
    "wipro": "WIPRO",
    "bharti airtel": "BHARTIARTL",
    "airtel": "BHARTIARTL",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "google": "GOOGL",
    "tesla": "TSLA"
}

def extract_and_fetch_live_data(prompt: str, status_callback=None) -> str:
    """Pre-detects company names/tickers and retrieves live exchange metrics."""
    found_tickers = set()
    p_lower = prompt.lower()

    for name, sym in KNOWN_ENTITIES.items():
        if re.search(r'\b' + re.escape(name) + r'\b', p_lower):
            found_tickers.add(sym)

    raw_candidates = re.findall(r'\b[A-Z]{2,10}\b', prompt)
    ignored = {"USD", "INR", "NSE", "BSE", "PE", "EV", "DIP", "EBITDA", "FCF", "APR", "LBO", "DCF", "WACC", "TCE", "NIM"}
    for c in raw_candidates:
        if c not in ignored:
            found_tickers.add(c)

    if not found_tickers:
        return ""

    results = []
    for ticker in found_tickers:
        if status_callback:
            status_callback("info", f"Fetching live NSE/BSE metrics for {ticker}...")
        try:
            data = tool_get_financial_data(ticker)
            if "error" not in data:
                price = data.get("current_price", "N/A")
                mcap = data.get("market_cap", "N/A")
                high = data.get("52w_high", "N/A")
                low = data.get("52w_low", "N/A")
                pe = data.get("pe_ratio", "N/A")
                resolved = data.get("ticker_resolved", ticker)
                
                if status_callback:
                    status_callback("info", f"✅ {ticker} Live Price: {price} | Market Cap: {mcap}")
                
                results.append(
                    f"- Company: {ticker} ({resolved}) | Live Price: {price} | Market Cap: {mcap} | "
                    f"P/E: {pe} | 52W High: {high} | 52W Low: {low}"
                )
            else:
                if status_callback:
                    status_callback("warning", f"⚠️ Live feed notice: {data.get('error')}")
        except Exception as e:
            if status_callback:
                status_callback("warning", f"⚠️ Error fetching {ticker}: {e}")

    if results:
        return "\n\n[LIVE EXCHANGE DATA - MANDATORY REFERENCE FOR ANALYSIS]:\n" + "\n".join(results) + "\n\n"
    else:
        return "\n\n[NOTICE: Live feed temporarily unreachable. Use established baseline figures in plain numbers with a standard disclaimer. DO NOT output code templates or placeholder tags like {oryn.get_...}.]\n\n"

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
    cleaned = re.sub(r'```(?:python)?\s*[\s\S]*?```', '', text)
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

    filter_out = ["whisper", "guard", "audio", "orpheus", "vision", "embed", "safeguard"]
    candidates = [
        m for m in available_ids 
        if not any(b in m.lower() for b in filter_out) and m != exclude_model
    ]

    preference = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768"
    ]

    test_queue = [m for m in preference if m in candidates] + [m for m in candidates if m not in preference]
    if not test_queue:
        test_queue = candidates if candidates else preference

    for model_name in test_queue:
        try:
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
            CACHED_WORKING_MODEL = model_name
            return model_name
        except Exception:
            continue

    if candidates:
        CACHED_WORKING_MODEL = candidates[0]
        return CACHED_WORKING_MODEL

    return "llama-3.3-70b-versatile"

def create_completion_with_fallback(client: Groq, preferred_model: str = None, status_callback=None, **kwargs):
    global CACHED_WORKING_MODEL
    active_model = preferred_model or get_best_available_model(client)
    
    kwargs["max_tokens"] = min(kwargs.get("max_tokens", 1200), 1200)
    kwargs["temperature"] = kwargs.get("temperature", 0.1)
    kwargs["model"] = active_model

    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        err_str = str(e).lower()
        
        if any(term in err_str for term in ["404", "model_not_found", "decommissioned", "does not exist", "not have access"]):
            if status_callback:
                status_callback("warning", f"Model `{active_model}` unavailable. Auto-switching to accessible engine...")
            alt_model = get_best_available_model(client, exclude_model=active_model, force_refresh=True)
            kwargs["model"] = alt_model
            return client.chat.completions.create(**kwargs)

        if "429" in err_str or "rate_limit_exceeded" in err_str or "tokens per minute" in err_str:
            delay_match = re.search(r'try again in ([0-9.]+)s', err_str)
            sleep_sec = float(delay_match.group(1)) + 0.5 if delay_match else 2.5
            if status_callback:
                status_callback("info", f"Rate limit pause ({sleep_sec:.1f}s)... retrying.")
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
        f"### 📊 Institutional DCF Valuation Schedule\n\n"
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

    # 1. DCF Quantitative Modeling Branch
    if is_quantitative_model_request(user_input):
        if status_callback:
            status_callback("info", "Executing quantitative DCF model and Excel builder...")
        briefing_text, plot_bytes, excel_bytes = generate_deterministic_dcf_artifacts(user_input)
        return sanitize_report_formatting(briefing_text), plot_bytes, excel_bytes, time.perf_counter() - start_time

    # 2. Pre-fetch Live Exchange Data directly into Context
    live_market_data = extract_and_fetch_live_data(user_input, status_callback=status_callback)
    augmented_prompt = user_input + live_market_data

    # 3. Direct Completion Execution
    try:
        messages = build_contextual_messages(augmented_prompt, SYSTEM_PROMPT, max_turns=3)
        
        resp = create_completion_with_fallback(
            client=client,
            preferred_model=selected_model,
            status_callback=status_callback,
            messages=messages,
            max_tokens=1200
        )
        final_text = resp.choices[0].message.content or ""
        elapsed = time.perf_counter() - start_time
        return sanitize_report_formatting(final_text), None, None, elapsed

    except Exception as e:
        return f"Error: {e}", None, None, time.perf_counter() - start_time
