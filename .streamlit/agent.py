import io
import json
import os
import re
import sys
import time
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq

# Safely import functions from tools with module reload protection
try:
    if "tools" in sys.modules:
        importlib.reload(sys.modules["tools"])
    import tools
    tool_get_financial_data = getattr(tools, "tool_get_financial_data", lambda x: {"error": "unavailable"})
    dynamic_web_search = getattr(tools, "dynamic_web_search", lambda x: None)
except Exception:
    def tool_get_financial_data(x):
        return {"error": "unavailable"}
    def dynamic_web_search(x):
        return None

# Blacklist of macro, ALM, FIG, and restructuring terms to prevent false-positive ticker lookups
FINANCIAL_ACRONYMS_BLACKLIST = {
    "NIM", "ALM", "TCE", "AFS", "FIG", "BPS", "DIP", "LBO", "WACC", "DCF",
    "EBITDA", "FCF", "APR", "EV", "PE", "IRR", "CAGR", "ROE", "ROA", "ROIC",
    "EPS", "NAV", "PBR", "PER", "DPS", "RBI", "SEBI", "FED", "CEO", "CFO",
    "CAN", "YOU", "PULL", "UP", "LIVE", "MARKET", "DATA", "FOR", "AND", "SHOW",
    "THEIR", "IN", "CLEAN", "TABLE", "WHAT", "WHO", "IS", "THE", "GIVE", "GET",
    "ME", "OF", "WITH", "THIS", "THAT", "FROM", "OUT", "RUN", "HOW", "ABOUT",
    "NOW", "ESTIMATE", "PRICE", "CAP", "RANGE", "RATIO", "STOCK", "SHARE",
    "USD", "INR", "NSE", "BSE", "FOUNDER", "FOUNDERS", "OF"
}

SYSTEM_PROMPT = (
    "You are ORYN Analyst, an elite institutional BFSI quantitative analyst and corporate intelligence advisor.\n\n"
    "CRITICAL CONTEXT & FACT PROTOCOL:\n"
    "- When [VERIFIED LIVE EXCHANGE DATA] or [VERIFIED RESEARCH RECORDS] is provided in the prompt, "
    "incorporate those verified facts into your response. Never contradict verified exchange quotes.\n"
    "- For leadership, officeholders, founders, or encyclopedic facts: if the search records do not explicitly state "
    "the answer, answer using your established factual baseline knowledge directly without refusing.\n\n"
    "CRITICAL FORMATTING & TABLE RULES (IMMUNITY TO KATEX COLLAPSE):\n"
    "1. DIRECT STRUCTURAL DELIVERY: Eliminate introductory pleasantries. Begin directly with the analysis header.\n"
    "2. NO DOLLAR SIGNS IN TABLES: NEVER use '$' inside table cells or column headers. "
    "State currency in headers as 'USD M' or 'INR' (e.g., | Year | Revenue (USD M) | EBITDA (USD M) |). "
    "Cell values MUST be pure numbers (e.g., | 1 | 60.00 | 15.00 |).\n"
    "3. STRICT MARKDOWN TABLE SYNTAX: Every table row MUST start and end with a pipe (|). "
    "Always place the divider row (| :--- | :--- |) immediately below the header row. Never output raw HTML table tags.\n\n"
    "FINANCIAL REASONING & WATERFALL PROTOCOLS:\n"
    "1. CHAPTER 11 RESTRUCTURING WATERFALL:\n"
    "   - Calculate claims strictly in linear sequence: DIP -> First Lien -> Second Lien -> Senior Unsecured -> Subordinated.\n"
    "   - Present the distribution in ONE unified Markdown table. Never debate or recalculate the table in subsequent text.\n"
    "   - The fulcrum security is the tranche that receives a partial recovery.\n"
    "2. FIG & ALM BANKING DYNAMICS:\n"
    "   - A policy rate cut on an asset-sensitive bank causes Net Interest Margin (NIM) COMPRESSION because floating assets reprice down faster than liabilities.\n"
    "   - High deposit beta cushions NIM compression (deposit costs drop quickly). Low deposit beta causes severe NIM compression.\n"
    "   - Falling benchmark yields increase bond prices, reversing unrealized losses in Available-For-Sale (AFS) portfolios, which directly EXPANDS Tangible Common Equity (TCE).\n"
    "3. DCF MODELING: Execute quantitative schedules, render matplotlib artifacts, and write 'model.xlsx'."
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
    injected_context = []
    p_lower = prompt.lower()

    # 1. Intent-Gated Equity Discovery
    found_tickers = set()
    for name, sym in KNOWN_ENTITIES.items():
        if re.search(r'\b' + re.escape(name) + r'\b', p_lower):
            found_tickers.add(sym)

    dollar_tickers = re.findall(r'\$([A-Za-z]{2,10})\b', prompt)
    for dt in dollar_tickers:
        dt_clean = dt.upper()
        if dt_clean not in FINANCIAL_ACRONYMS_BLACKLIST:
            found_tickers.add(dt_clean)

    is_stock_request = any(k in p_lower for k in ["stock", "share", "price", "market cap", "52-week", "quote", "ticker"])
    if is_stock_request:
        raw_candidates = re.findall(r'\b[A-Z]{2,10}\b', prompt)
        for c in raw_candidates:
            if c not in FINANCIAL_ACRONYMS_BLACKLIST:
                found_tickers.add(c)

    target_tickers = list(found_tickers)[:4]

    if target_tickers:
        market_rows = []
        for ticker in target_tickers:
            if status_callback:
                status_callback("info", f"Retrieving verified market metrics for {ticker}...")
            try:
                data = tool_get_financial_data(ticker)
                price = data.get("current_price", "N/A")
                mcap = data.get("market_cap", "N/A")
                high = data.get("52w_high", "N/A")
                low = data.get("52w_low", "N/A")
                resolved = data.get("ticker_resolved", ticker)
                cname = data.get("company_name", ticker)

                if status_callback and price != "N/A":
                    status_callback("info", f"✅ {ticker} Live Price: {price} | Market Cap: {mcap}")

                market_rows.append(
                    f"- {cname} ({resolved}): Current Price = {price}, Market Cap = {mcap}, "
                    f"52-Week High = {high}, 52-Week Low = {low}"
                )
            except Exception:
                continue

        if market_rows:
            injected_context.append(
                "[VERIFIED LIVE EXCHANGE DATA - USE IN TABLE]:\n" + 
                "\n".join(market_rows)
            )

    # 2. General Knowledge / Leadership / Corporate Intelligence Research
    is_pure_stock_quote = bool(target_tickers) and is_stock_request
    if not is_pure_stock_quote:
        if not any(w in p_lower for w in ["dcf", "discounted cash flow", "wacc", "waterfall", "restructuring"]):
            if status_callback:
                status_callback("info", "Autonomous research: retrieving verified records...")
            scraped_intel = dynamic_web_search(prompt)
            if scraped_intel:
                injected_context.append(
                    f"[VERIFIED RESEARCH RECORDS]:\n{scraped_intel}"
                )

    if injected_context:
        return "\n\n" + "\n\n".join(injected_context) + "\n\n"
    return ""

def sanitize_report_formatting(text: str) -> str:
    """Repairs tables, strips list prefixes, and immunizes output against KaTeX math mode."""
    cleaned = re.sub(r'```(?:python)?\s*[\s\S]*?```', '', text)
    cleaned = re.sub(r'\|\s*\|', '|\n|', cleaned)

    lines = cleaned.split('\n')
    output = []
    current_table = []

    def flush_tbl(tbl):
        if not tbl:
            return []
        if len(tbl) >= 2 and not any('---' in r for r in tbl[:2]):
            header_cols = [c.strip() for c in tbl[0].strip('|').split('|')]
            divider = "| " + " | ".join([":---"] * len(header_cols)) + " |"
            tbl.insert(1, divider)
        return [""] + tbl + [""]

    for line in lines:
        stripped = line.strip()
        stripped = re.sub(r'^\d+[\.\)]\s*\|', '|', stripped)
        
        if '|' in stripped:
            if not stripped.startswith('|'):
                stripped = '| ' + stripped
            if not stripped.endswith('|'):
                stripped = stripped + ' |'
            stripped = stripped.replace('$', 'USD ')
            current_table.append(stripped)
        else:
            if current_table:
                output.extend(flush_tbl(current_table))
                current_table = []
            output.append(line)

    if current_table:
        output.extend(flush_tbl(current_table))

    res = '\n'.join(output)
    res = re.sub(r'\$\s*([0-9])', r'USD \1', res)
    return re.sub(r'\n{3,}', '\n\n', res).strip()

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
    candidates = [m for m in available_ids if not any(b in m.lower() for b in filter_out) and m != exclude_model]
    preference = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "llama3-8b-8192"]

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

    return "llama-3.3-70b-versatile"

def create_completion_with_fallback(client: Groq, preferred_model: str = None, status_callback=None, **kwargs):
    global CACHED_WORKING_MODEL
    active_model = preferred_model or get_best_available_model(client)
    
    kwargs["max_tokens"] = min(kwargs.get("max_tokens", 800), 800)
    kwargs["temperature"] = kwargs.get("temperature", 0.1)
    kwargs["frequency_penalty"] = 0.15
    kwargs["presence_penalty"] = 0.1
    kwargs["model"] = active_model

    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        err_str = str(e).lower()

        if any(term in err_str for term in ["output tokens per minute", "otpm", "reduce max_tokens", "rate_limit_exceeded"]):
            if kwargs["max_tokens"] > 500:
                kwargs["max_tokens"] = 500
                if status_callback:
                    status_callback("info", "Adjusting output budget to fit tier quota... retrying.")
                return client.chat.completions.create(**kwargs)

        if any(term in err_str for term in ["404", "model_not_found", "decommissioned", "does not exist", "not have access"]):
            alt_model = get_best_available_model(client, exclude_model=active_model, force_refresh=True)
            kwargs["model"] = alt_model
            return client.chat.completions.create(**kwargs)

        if "429" in err_str:
            delay_match = re.search(r'try again in ([0-9.]+)s', err_str)
            sleep_sec = float(delay_match.group(1)) + 0.5 if delay_match else 2.5
            time.sleep(sleep_sec)
            alt_model = get_best_available_model(client, exclude_model=active_model, force_refresh=True)
            kwargs["model"] = alt_model
            return client.chat.completions.create(**kwargs)
        raise e

def build_contextual_messages(user_input: str, system_prompt: str, max_turns: int = 2) -> list[dict]:
    history = []
    try:
        import streamlit as st
        if "messages" in st.session_state and isinstance(st.session_state.messages, list):
            candidates = st.session_state.messages[:-1] if st.session_state.messages[-1].get("content") == user_input else st.session_state.messages
            for m in candidates[-(max_turns * 2):]:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role in ["user", "assistant"] and content:
                    history.append({"role": role, "content": str(content)})
    except Exception:
        pass
    return [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_input}]

def parse_dcf_assumptions(prompt: str) -> dict:
    """Parses dynamic DCF parameters from conversational input with fallback defaults."""
    p_lower = prompt.lower()
    
    # 1. Growth rate (default 20%)
    growth_match = re.search(r'(?:growing|growth rate|revenue growth|growth)[\s:]*(?:at\s*)?([0-9.]+)%', p_lower)
    growth = float(growth_match.group(1)) / 100.0 if growth_match else 0.20

    # 2. Revenue Parsing
    y1_num = None
    m1 = re.search(r'([0-9.]+)\s*(?:m|million)?\s*(?:revenue\s+)?in\s+year[\s-]*1', p_lower)
    m2 = re.search(r'rev(?:enue)?[\s:]*(?:usd|\$)?\s*([0-9.]+)\s*(?:m|million)?\s+in\s+year[\s-]*1', p_lower)
    m3 = re.search(r'year[\s-]*1\s*rev(?:enue)?[\s:]*(?:usd|\$)?\s*([0-9.]+)', p_lower)

    if m1:
        y1_num = float(m1.group(1))
    elif m2:
        y1_num = float(m2.group(1))
    elif m3:
        y1_num = float(m3.group(1))

    if y1_num is not None:
        base_revenue = y1_num / (1.0 + growth) if growth > -0.99 else y1_num
    else:
        m_gen = re.search(r'(?:revenue|rev)[\s:]*(?:usd|\$)?\s*([0-9.]+)', p_lower)
        if m_gen:
            val = float(m_gen.group(1))
            if "start" in p_lower or "base" in p_lower:
                base_revenue = val
            else:
                base_revenue = val / (1.0 + growth)
        else:
            base_revenue = 50.0

    # 3. EBITDA Margin
    margin_match = re.search(r'(?:ebitda margin|margin|ebitda)[\s:]*(?:apparently\s*)?([0-9.]+)%', p_lower)
    margin = float(margin_match.group(1)) / 100.0 if margin_match else 0.25

    # 4. WACC / Discount Rate
    wacc_match = re.search(r'(?:wacc|discount rate|cost of capital)[\s:]*(?:apparently\s*)?([0-9.]+)%', p_lower)
    wacc = float(wacc_match.group(1)) / 100.0 if wacc_match else 0.10

    # 5. Terminal Growth Rate
    tv_match = re.search(r'(?:terminal growth|terminal rate|perpetual growth)[\s:]*(?:apparently\s*)?([0-9.]+)%', p_lower)
    tv_rate = float(tv_match.group(1)) / 100.0 if tv_match else 0.025

    return {
        "base_revenue": base_revenue,
        "growth": growth,
        "margin": margin,
        "wacc": wacc,
        "tv_rate": tv_rate
    }

def generate_deterministic_dcf_artifacts(user_input: str) -> tuple[str, bytes | None, bytes | None]:
    params = parse_dcf_assumptions(user_input)
    base_rev = params["base_revenue"]
    growth = params["growth"]
    margin = params["margin"]
    wacc = params["wacc"]
    tv_rate = params["tv_rate"]
    years = 5

    # Gordon Growth Model validation: terminal growth must be strictly lower than discount rate
    if tv_rate >= wacc:
        err_msg = (
            f"⚠️ **DCF Model Exception**: Terminal growth rate ({tv_rate:.1%}) must be strictly less than "
            f"WACC ({wacc:.1%}) to avoid a mathematical divergence in the Gordon Growth calculation."
        )
        return err_msg, None, None

    years_idx = list(range(1, years + 1))
    rev = [base_rev * ((1 + growth) ** t) for t in years_idx]
    ebitda = [r * margin for r in rev]
    df = [1 / ((1 + wacc) ** t) for t in years_idx]
    
    # Standard UFCF modeling convention
    pv_ufcf = [u * d for u, d in zip(ebitda, df)]
    sum_pv_ufcf = sum(pv_ufcf)
    tv = (ebitda[-1] * (1 + tv_rate)) / (wacc - tv_rate)
    pv_tv = tv / ((1 + wacc) ** years)
    ev = sum_pv_ufcf + pv_tv

    df_proj = pd.DataFrame({
        "Year": [str(y) for y in years_idx],
        "Revenue (USD M)": [f"{x:.2f}" for x in rev],
        "EBITDA (USD M)": [f"{x:.2f}" for x in ebitda],
        "Discount Factor": [f"{x:.4f}" for x in df],
        "PV of UFCF (USD M)": [f"{x:.2f}" for x in pv_ufcf]
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(['PV Cash Flows', 'PV Terminal Value'], [sum_pv_ufcf, pv_tv], color=['#4a148c', '#7b1fa2'], width=0.45)
    ax.set_title(f'Enterprise Value: USD {ev:.2f}M', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=200)
    plt.close('all')

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
        df_proj.to_excel(writer, sheet_name='DCF_Projections', index=False)

    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df_proj.values]
    headers = "| " + " | ".join(df_proj.columns) + " |"
    dividers = "| " + " | ".join([":---"] * len(df_proj.columns)) + " |"
    table_md = "\n".join([headers, dividers] + rows)

    calc_briefing = (
        f"### 📊 Institutional DCF Valuation Schedule\n\n"
        f"**Implied Enterprise Value:** **USD {ev:.2f}M**\n\n"
        f"{table_md}\n\n"
        f"* **Model Assumptions:** Year-1 Rev: USD {rev[0]:.2f}M • Growth: {growth:.1%} • Margin: {margin:.1%} • WACC: {wacc:.1%} • Terminal Rate: {tv_rate:.1%}\n"
        f"* **Cumulative PV of Cash Flows:** USD {sum_pv_ufcf:.2f}M\n"
        f"* **Terminal Value (TV):** USD {tv:.2f}M\n"
        f"* **Present Value of TV:** USD {pv_tv:.2f}M\n"
        f"* **Implied Enterprise Value (EV):** **USD {ev:.2f}M**"
    )
    return calc_briefing, img_buf.getvalue(), excel_buf.getvalue()

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

    if is_quantitative_model_request(user_input):
        briefing_text, plot_bytes, excel_bytes = generate_deterministic_dcf_artifacts(user_input)
        return sanitize_report_formatting(briefing_text), plot_bytes, excel_bytes, time.perf_counter() - start_time

    context_data = extract_and_fetch_live_data(user_input, status_callback=status_callback)
    augmented_prompt = user_input + context_data

    try:
        messages = build_contextual_messages(augmented_prompt, SYSTEM_PROMPT, max_turns=2)
        resp = create_completion_with_fallback(
            client=client,
            preferred_model=selected_model,
            status_callback=status_callback,
            messages=messages,
            max_tokens=800
        )
        final_text = resp.choices[0].message.content or ""
        elapsed = time.perf_counter() - start_time
        return sanitize_report_formatting(final_text), None, None, elapsed
    except Exception as e:
        return f"Error: {e}", None, None, time.perf_counter() - start_time
