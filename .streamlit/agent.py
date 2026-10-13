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
    "USD", "INR", "NSE", "BSE", "FOUNDER", "FOUNDERS", "OF",
    "CURRENT", "TODAY", "LATEST", "PLEASE", "CHECK", "TELL", "STOCKS", "COMPANY",
    "COMPARE", "ANALYZE", "ANALYSE", "WHICH", "WHEN", "WHERE", "WHY", "ARE",
    "WAS", "WERE", "DO", "DOES", "COULD", "WOULD", "SHOULD", "NEED", "OPEN",
    "CLOSE", "CLOSING", "HIGH", "LOW", "DAY", "WEEK", "MONTH"
}

SYSTEM_PROMPT = (
    "You are ORYN Analyst, an elite institutional BFSI quantitative analyst and corporate intelligence advisor.\n\n"
    "CRITICAL FACT PROTOCOL:\n"
    "- Treat retrieved market data and research snippets as evidence, not guaranteed truth. "
    "State source and retrieval timestamp when available; never label data verified unless metadata supports that claim.\n"
    "- If asked who founded a company or holds public office, state the exact names and dates directly from the verified record.\n\n"
    "CRITICAL FORMATTING & TABLE RULES (IMMUNITY TO KATEX COLLAPSE):\n"
    "1. DIRECT STRUCTURAL DELIVERY: Eliminate introductory pleasantries. Begin directly with the analysis header.\n"
    "2. NO DOLLAR SIGNS IN TABLES: NEVER use '$' inside table cells or column headers. "
    "State currency in headers as 'USD M' or 'INR' (e.g., | Security Class | Total Claim (USD M) | Recovery (USD M) |). "
    "Cell values MUST be pure numbers (e.g., | DIP Financing | 50.00 | 50.00 | 100.0% |).\n"
    "3. STRICT MARKDOWN TABLE SYNTAX: Every table row MUST start and end with a pipe (|). "
    "Always place the divider row (| :--- | :--- |) immediately below the header row. Never output raw HTML table tags.\n\n"
    "FINANCIAL REASONING & WATERFALL PROTOCOLS:\n"
    "1. CHAPTER 11 RESTRUCTURING WATERFALL:\n"
    "   - Calculate claims strictly in linear priority order: DIP -> First Lien -> Second Lien -> Senior Unsecured -> Subordinated.\n"
    "   - Tranche recovery = min(Claim, max(0, Remaining EV before tranche)).\n"
    "   - Present the distribution in ONE unified Markdown table. Never debate, recalculate, or re-evaluate the table in subsequent paragraphs.\n"
    "   - The fulcrum security is the tranche that receives a partial recovery.\n"
    "2. FIG & ALM BANKING DYNAMICS:\n"
    "   - A policy rate cut on an asset-sensitive bank causes Net Interest Margin (NIM) COMPRESSION because floating assets reprice downward faster than liabilities.\n"
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

    # Trigger stock search only if prompt explicitly requests equity metrics
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
                status_callback("info", f"Retrieving market data for {ticker}...")
            try:
                data = tool_get_financial_data(ticker)
                price = data.get("current_price", "N/A")
                mcap = data.get("market_cap", "N/A")
                high = data.get("52w_high", "N/A")
                low = data.get("52w_low", "N/A")
                resolved = data.get("ticker_resolved", ticker)
                cname = data.get("company_name", ticker)

                if status_callback and price != "N/A":
                    status_callback("info", f"Retrieved {ticker} price: {price} | Source: {data.get('source', 'unknown')} | Retrieved UTC: {data.get('retrieved_at_utc', 'unknown')}")

                market_rows.append(
                    f"- {cname} ({resolved}): Price = {price}, Market Cap = {mcap}, "
                    f"52-Week High = {high}, 52-Week Low = {low}; "
                    f"Source = {data.get('source', 'unknown')}; Retrieved UTC = {data.get('retrieved_at_utc', 'unknown')}"
                )
            except Exception:
                continue

        if market_rows:
            injected_context.append(
                "[RETRIEVED MARKET DATA - CHECK SOURCE AND TIMESTAMP]:\n" + 
                "\n".join(market_rows)
            )

    # 2. General Knowledge / Leadership / Corporate Intelligence Research
    is_pure_stock_quote = bool(target_tickers) and is_stock_request
    if not is_pure_stock_quote:
        # Avoid running search for pure quantitative modeling or restructuring prompts
        if not any(w in p_lower for w in ["dcf", "discounted cash flow", "wacc", "waterfall", "restructuring", "ebitda margin"]):
            if status_callback:
                status_callback("info", "Researching available sources...")
            scraped_intel = dynamic_web_search(prompt)
            if scraped_intel:
                injected_context.append(
                    f"[RETRIEVED RESEARCH SNIPPETS - CHECK SOURCES BEFORE ASSERTING FACTS]:\n{scraped_intel}"
                )

    if injected_context:
        return "\n\n" + "\n\n".join(injected_context) + "\n\n"
    return ""

def sanitize_report_formatting(text: str) -> str:
    """Repairs tables, strips list prefixes, and immunizes output against KaTeX math mode."""
    code_blocks = []
    def preserve_code(match):
        code_blocks.append(match.group(0))
        return f"ORYN_CODE_BLOCK_{len(code_blocks) - 1}_PLACEHOLDER"
    cleaned = re.sub(r'```[\s\S]*?```', preserve_code, text)
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
        # Strip leading ordered list numbers (e.g., '1. | Company |' -> '| Company |')
        stripped = re.sub(r'^\d+[\.\)]\s*\|', '|', stripped)
        
        if '|' in stripped:
            if not stripped.startswith('|'):
                stripped = '| ' + stripped
            if not stripped.endswith('|'):
                stripped = stripped + ' |'
            # Immunize table row from KaTeX: replace '$' inside tables with 'USD '
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
    # Immunize body text: replace all '$' followed by numbers with 'USD '
    res = re.sub(r'\$\s*([0-9])', r'USD \1', res)
    for index, block in enumerate(code_blocks):
        res = res.replace(f"ORYN_CODE_BLOCK_{index}_PLACEHOLDER", block)
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
    # Anti-repetition penalties prevent infinite loops
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

def _percent_assumption(prompt: str, patterns: list[str], default: float) -> float:
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)) / 100.0
    return default


def generate_deterministic_dcf_artifacts(user_input: str) -> tuple[str, bytes | None, bytes | None]:
    """Build an illustrative DCF with transparent UFCF components and disclosed defaults."""
    growth = _percent_assumption(user_input, [
        r"(?:revenue\s+growth|growth\s+rate)\s*(?:of|is|at|=|:)?\s*([\d.]+)\s*%",
        r"([\d.]+)\s*%\s*(?:annual\s+)?revenue\s+growth",
        r"(?:growing|grows|grow)\s+(?:at\s+)?([\d.]+)\s*%"
    ], 0.20)
    base_match = re.search(r"(?:starting|base|initial)\s+revenue(?:\s+of|\s*[:=])?\s*\$?\s*([\d,.]+)\s*(billion|bn|million|mn|m|b)?", user_input, re.I)
    year1_match = re.search(r"revenue\s*:\s*(?:usd\s*)?\$?\s*([\d,.]+)\s*(billion|bn|million|mn|m|b)?\s+in\s+year\s*1", user_input, re.I)
    def to_millions(match):
        value = float(match.group(1).replace(',', ''))
        return value * 1000 if (match.group(2) or '').lower() in {'billion', 'bn', 'b'} else value
    if base_match:
        base_revenue = to_millions(base_match)
        revenue_basis = 'Year-0 starting revenue'
    elif year1_match:
        year1_revenue = to_millions(year1_match)
        base_revenue = year1_revenue / (1 + growth)
        revenue_basis = f'Year-1 revenue of USD {year1_revenue:.2f}M (Year-0 base back-solved)'
    else:
        base_revenue, revenue_basis = 50.0, 'default Year-0 starting revenue'
    ebitda_margin = _percent_assumption(user_input, [r"(?:ebitda\s+margin|margin)\s*(?:of|is|at|=|:)?\s*(?:apparently\s+)?([\d.]+)\s*%", r"([\d.]+)\s*%\s*ebitda\s+margin"], 0.25)
    wacc = _percent_assumption(user_input, [r"(?:wacc|discount\s+rate)\s*(?:of|is|at|=|:)?\s*(?:apparently\s+)?([\d.]+)\s*%", r"([\d.]+)\s*%\s*(?:wacc|discount\s+rate)"], 0.10)
    terminal_growth = _percent_assumption(user_input, [r"(?:terminal\s+growth|perpetual\s+growth)\s*(?:rate)?\s*(?:of|is|at|=|:)?\s*(?:apparently\s+)?([\d.]+)\s*%", r"([\d.]+)\s*%\s*(?:terminal|perpetual)\s+growth"], 0.025)
    if base_revenue <= 0 or growth <= -1 or not 0 < ebitda_margin <= 1:
        return 'Invalid DCF assumptions: revenue must be positive, growth must exceed -100%, and EBITDA margin must be greater than 0% and at most 100%.', None, None
    if not 0 < wacc < 1 or not 0 <= terminal_growth < wacc:
        return 'Invalid DCF assumptions: WACC must be between 0% and 100%, and terminal growth must be non-negative and lower than WACC.', None, None
    years = list(range(1, 6))
    tax_rate, da_pct, capex_pct, nwc_pct = 0.25, 0.03, 0.04, 0.01
    revenue = [base_revenue * (1 + growth) ** y for y in years]
    ebitda = [x * ebitda_margin for x in revenue]
    da = [x * da_pct for x in revenue]
    ebit = [a - b for a, b in zip(ebitda, da)]
    taxes = [max(0, x * tax_rate) for x in ebit]
    capex = [x * capex_pct for x in revenue]
    dnwc = [(revenue[i] - (base_revenue if i == 0 else revenue[i-1])) * nwc_pct for i in range(5)]
    ufcf = [e - t + d - c - n for e, t, d, c, n in zip(ebit, taxes, da, capex, dnwc)]
    factors = [1 / (1 + wacc) ** y for y in years]
    pv = [x * f for x, f in zip(ufcf, factors)]
    pv_cashflows = sum(pv)
    terminal_value = ufcf[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** 5
    ev = pv_cashflows + pv_terminal
    df_proj = pd.DataFrame({'Year': years, 'Revenue (USD M)': [round(x,2) for x in revenue], 'EBITDA (USD M)': [round(x,2) for x in ebitda], 'D&A (USD M)': [round(x,2) for x in da], 'EBIT (USD M)': [round(x,2) for x in ebit], 'Cash Taxes (USD M)': [round(x,2) for x in taxes], 'CapEx (USD M)': [round(x,2) for x in capex], 'Change in NWC (USD M)': [round(x,2) for x in dnwc], 'UFCF (USD M)': [round(x,2) for x in ufcf], 'Discount Factor': [round(x,4) for x in factors], 'PV of UFCF (USD M)': [round(x,2) for x in pv]})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(['PV of UFCF', 'PV of Terminal Value'], [pv_cashflows, pv_terminal], width=0.45)
    ax.set_title(f'Illustrative Enterprise Value: USD {ev:.2f}M')
    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=160)
    plt.close('all')
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
        df_proj.to_excel(writer, sheet_name='DCF_Projections', index=False)
    headers = '| ' + ' | '.join(df_proj.columns) + ' |'
    divider = '| ' + ' | '.join([':---'] * len(df_proj.columns)) + ' |'
    rows = ['| ' + ' | '.join(map(str, row)) + ' |' for row in df_proj.values]
    table = '\n'.join([headers, divider] + rows)
    assumptions = (f'- Revenue basis: {revenue_basis}\n- Year-0 starting revenue: USD {base_revenue:.2f}M\n- Revenue growth: {growth:.2%}\n- EBITDA margin: {ebitda_margin:.2%}\n- WACC: {wacc:.2%}\n- Terminal growth: {terminal_growth:.2%}\n- Illustrative defaults: tax {tax_rate:.0%}, D&A {da_pct:.0%} of revenue, CapEx {capex_pct:.0%} of revenue, change in NWC {nwc_pct:.0%} of incremental revenue.')
    report = (f'### Illustrative DCF Valuation\n\n**Implied Enterprise Value: USD {ev:.2f}M**\n\n**Important:** This is not a verified valuation of Reliance Industries. Validate all inputs against company financial statements.\n\n**Assumptions**\n{assumptions}\n\n{table}\n\n- Present value of projected UFCF: USD {pv_cashflows:.2f}M\n- Terminal value: USD {terminal_value:.2f}M\n- Present value of terminal value: USD {pv_terminal:.2f}M\n- Enterprise value = PV of projected UFCF + PV of terminal value.')
    return report, img_buf.getvalue(), excel_buf.getvalue()
def run_analyst_agent(
    user_input: str,
    api_key: str = "",
    selected_model: str = None,
    hitl_enabled: bool = True,
    status_callback=None,
    **kwargs
) -> tuple[str, bytes | None, bytes | None, float]:
    start_time = time.perf_counter()
    if is_quantitative_model_request(user_input):
        briefing_text, plot_bytes, excel_bytes = generate_deterministic_dcf_artifacts(user_input)
        return sanitize_report_formatting(briefing_text), plot_bytes, excel_bytes, time.perf_counter() - start_time

    active_key = resolve_api_key(api_key)
    if not active_key:
        return "⚠️ **API Key Missing**: Please provide a Groq API key in the sidebar.", None, None, 0.0

    try:
        client = Groq(api_key=active_key, max_retries=0)
    except Exception as e:
        return f"⚠️ **Client Error**: {e}", None, None, 0.0

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
