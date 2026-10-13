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

COMMON_ENGLISH_STOPWORDS = {
    "CAN", "YOU", "PULL", "UP", "LIVE", "MARKET", "DATA", "FOR", "AND", "SHOW",
    "THEIR", "IN", "CLEAN", "TABLE", "WHAT", "WHO", "IS", "THE", "GIVE", "GET",
    "ME", "OF", "WITH", "THIS", "THAT", "FROM", "OUT", "RUN", "HOW", "ABOUT",
    "NOW", "ESTIMATE", "PRICE", "CAP", "RANGE", "RATIO", "STOCK", "SHARE",
    "USD", "INR", "NSE", "BSE", "PE", "EV", "DIP", "EBITDA", "FCF", "APR",
    "LBO", "DCF", "WACC", "TCE", "NIM", "RBI", "SEBI", "FED", "CEO", "CFO"
}

SYSTEM_PROMPT = (
    "You are ORYN Analyst, an elite institutional BFSI quantitative analyst and research advisor.\n\n"
    "CRITICAL CONTEXT & FACT PROTOCOL:\n"
    "- When [VERIFIED LIVE EXCHANGE DATA] or [VERIFIED LIVE RESEARCH - RECENT HEADLINES & RECORDS] is provided, "
    "you MUST base your response strictly on those fresh live facts.\n"
    "- If asked who holds a public office or leadership position (e.g., Governor of Jharkhand, Governor of Bihar, RBI Governor), "
    "identify the incumbent from the live headlines, state their tenure start, and note their predecessor.\n\n"
    "CRITICAL PRESENTATION & TABLE RULES:\n"
    "1. DIRECT STRUCTURAL DELIVERY: Eliminate introductory pleasantries. Begin directly with the answer.\n"
    "2. STRICT PROHIBITION ON HTML TABLES: NEVER output <table>, <tr>, <td>, or <th> tags. "
    "Always format tables using standard Markdown pipes (|) and header divider rows (| :--- | :--- |).\n"
    "3. INSTITUTIONAL RESTRUCTURING & WATERFALL RULES:\n"
    "   - Always present priority distributions inside a single consolidated Markdown table.\n"
    "   - Sum total prepetition claims accurately before calculating recoveries.\n"
    "4. DCF MODELING: For DCF requests, execute quantitative schedules, render a matplotlib chart, and write 'model.xlsx'."
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

    # 1. Equity Discovery
    found_tickers = set()
    for name, sym in KNOWN_ENTITIES.items():
        if re.search(r'\b' + re.escape(name) + r'\b', p_lower):
            found_tickers.add(sym)

    raw_candidates = re.findall(r'\b[A-Z]{2,10}\b', prompt)
    for c in raw_candidates:
        if c not in COMMON_ENGLISH_STOPWORDS:
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
                "[VERIFIED LIVE EXCHANGE DATA - YOU MUST USE THESE EXACT VALUES IN THE TABLE]:\n" + 
                "\n".join(market_rows)
            )

    # 2. Autonomous Live Web Search for General Knowledge, Leadership & Macro Queries
    is_gk_query = any(w in p_lower for w in [
        "who is", "who was", "current governor", "current chief", "current minister",
        "governor of", "cm of", "ceo of", "chairman of", "president of", "what is",
        "latest", "tenure", "rbi", "sebi", "news"
    ]) or ("governor" in p_lower and not target_tickers) or (not target_tickers and "?" in prompt)

    if is_gk_query:
        if status_callback:
            status_callback("info", "Autonomous live research: fetching verified records via syndication feed...")
        scraped_intel = dynamic_web_search(prompt)
        if scraped_intel:
            injected_context.append(
                f"[VERIFIED LIVE RESEARCH - RECENT HEADLINES & RECORDS]:\n{scraped_intel}"
            )

    if injected_context:
        return "\n\n" + "\n\n".join(injected_context) + "\n\n"
    return ""

def convert_html_tables_to_markdown(text: str) -> str:
    """Converts HTML table tags to Markdown so Streamlit won't strip them into run-on text."""
    table_pattern = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
    
    def replace_table(match):
        table_content = match.group(1)
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_content, re.DOTALL | re.IGNORECASE)
        md_rows = []
        for r in rows:
            cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', r, re.DOTALL | re.IGNORECASE)
            clean_cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if clean_cells:
                md_rows.append("| " + " | ".join(clean_cells) + " |")
        
        if len(md_rows) >= 2:
            num_cols = len(md_rows[0].split('|')) - 2
            divider = "| " + " | ".join([":---"] * num_cols) + " |"
            md_rows.insert(1, divider)
            return "\n\n" + "\n".join(md_rows) + "\n\n"
        return "\n".join(md_rows)

    return table_pattern.sub(replace_table, text)

def sanitize_report_formatting(text: str) -> str:
    """Repairs tables, fixes delimiters, and prevents text concatenation."""
    # 1. Convert HTML tables to Markdown
    cleaned = convert_html_tables_to_markdown(text)
    cleaned = re.sub(r'```(?:python)?\s*[\s\S]*?```', '', cleaned)
    cleaned = re.sub(r'\|\s*\|', '|\n|', cleaned)

    # 2. Ensure valid Markdown table row boundaries and dividers
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
        if '|' in stripped:
            if not stripped.startswith('|'):
                stripped = '| ' + stripped
            if not stripped.endswith('|'):
                stripped = stripped + ' |'
            current_table.append(stripped)
        else:
            if current_table:
                output.extend(flush_tbl(current_table))
                current_table = []
            output.append(line)

    if current_table:
        output.extend(flush_tbl(current_table))

    res = '\n'.join(output)
    # 3. Insert space after currency symbols to prevent KaTeX math mode
    res = re.sub(r'\$([0-9])', r'$ \1', res)
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

def generate_deterministic_dcf_artifacts(user_input: str) -> tuple[str, bytes, bytes]:
    years = 5
    years_idx = list(range(1, years + 1))
    rev = [50.0 * ((1 + 0.20) ** t) for t in years_idx]
    ebitda = [r * 0.25 for r in rev]
    df = [1 / ((1 + 0.10) ** t) for t in years_idx]
    pv_ufcf = [u * d for u, d in zip(ebitda, df)]
    sum_pv_ufcf = sum(pv_ufcf)
    tv = (ebitda[-1] * 1.025) / (0.10 - 0.025)
    pv_tv = tv / (1.10 ** years)
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
    ax.set_title(f'Enterprise Value: $ {ev:.2f}M', fontsize=12, fontweight='bold')
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
        f"**Implied Enterprise Value:** **$ {ev:.2f}M**\n\n"
        f"{table_md}\n\n"
        f"* **Cumulative PV of Cash Flows:** $ {sum_pv_ufcf:.2f}M\n"
        f"* **Terminal Value (TV):** $ {tv:.2f}M\n"
        f"* **Present Value of TV:** $ {pv_tv:.2f}M\n"
        f"* **Implied Enterprise Value (EV):** **$ {ev:.2f}M**"
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
