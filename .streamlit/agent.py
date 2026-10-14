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

# Safely import market data tools with fallback protection
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
    "USD", "INR", "NSE", "BSE", "FOUNDER", "FOUNDERS", "OF", "TSM", "MOIC",
    "CET1", "RWA", "LCR", "NSFR", "HQLA"
}

SYSTEM_PROMPT = (
    "You are ORYN Analyst, an elite institutional BFSI quantitative analyst and corporate intelligence advisor.\n\n"
    "CRITICAL CONVERSATION & MEMORY VAULT PROTOCOL:\n"
    "1. DOCUMENT-FIRST QUANTITATIVE RIGOR: When institutional filing content is provided under [OFFICIAL FINANCIAL FILING DOCUMENT ATTACHED], "
    "perform all quantitative calculations, covenant compliance tests, and balance sheet waterfalls DIRECTLY and IMMEDIATELY using the "
    "exact numbers, sections, and definitions in that filing text. Show complete, unhurried calculations for all requested parts. Do not stop until every part is answered.\n"
    "2. MULTI-TURN CONTINUITY ACROSS SESSIONS: Review ALL conversation turns and archived session history in your prompt. "
    "When a user asks to 're-run', 'sensitize', or 'modify' a previous calculation, retain ALL baseline parameters (EBITDA, purchase price, "
    "equity check, debt balances) from the prior prompt or session archive and ONLY adjust the requested variable.\n"
    "3. NO BRACKET TAGS OR ECHOING: NEVER output bracketed headers (e.g., [ORYN REPORT]). Begin directly with the analysis.\n"
    "4. NO CHATBOT HEDGING: Eliminate conversational filler (e.g., 'Based on the available information'). Deliver crisp intelligence.\n"
    "5. CLEAN MARKDOWN FORMATTING: Present formulas in plain, bold text (e.g., **Adjusted EBITDA = Unadjusted EBITDA + Capped Add-backs**). "
    "Do NOT output raw brackets like '[ \\text{...} ]' or isolated '$' symbols that disrupt markdown rendering.\n"
    "6. REAL-WORLD INCUMBENCY (STRICT ADHERENCE):\n"
    "   - Tamil Nadu: Chief Minister is M. K. Stalin (DMK), in office since May 2021.\n"
    "   - Maharashtra: Chief Minister is Devendra Fadnavis (BJP / Mahayuti), in office since December 2024.\n"
    "   - SBI: Chairman is Challa Sreenivasulu Setty (C.S. Setty), appointed August 2024.\n"
    "   - LIC: Executive head is CEO & MD R. Doraiswamy, appointed July 2025.\n"
    "   - SpaceX: Founded in 2002 by Elon Musk.\n\n"
    "FIG BANKING & CORPORATE FINANCE LOGIC:\n"
    "1. COMMERCIAL BANK COVENANTS & LEVERAGE:\n"
    "   - Regulated commercial banks DO NOT have 'Debt-to-EBITDA' covenants. Deposits and debt are funding liabilities, not corporate debt.\n"
    "   - Banks are governed by Basel III Regulatory Capital Requirements: CET1 ratio (>= 4.5%), Tier 1 ratio (>= 6.0%), "
    "Total Capital ratio (>= 8.0%), and the Basel III Leverage Ratio (Tier 1 Capital / Total Leverage Exposure >= 3.0%).\n"
    "2. ACCOUNTING DEFINITIONS:\n"
    "   - EBITDA is Earnings BEFORE Interest, Taxes, Depreciation, and Amortization. Interest expense NEVER alters or reduces EBITDA.\n"
    "3. VALUATION & RESTRUCTURING WATERFALLS:\n"
    "   - Equity Value = Implied EV - Net Debt - Minority Interest - Pension Deficit + Equity Investments in Affiliates.\n"
    "   - Net Debt = Total Gross Debt - Unrestricted Cash (exclude restricted cash).\n"
    "   - Chapter 11 priority: DIP -> First Lien -> Second Lien -> Senior Unsecured -> Subordinated. Fulcrum security is partially impaired.\n"
    "   - Treasury Stock Method (TSM): Proceeds = Options * Strike Price; Repurchased Shares = Proceeds / Share Price; "
    "Net Dilutive Shares = Options - Repurchased; Fully Diluted = Basic + Net Dilutive.\n"
    "   - LBO Returns: MOIC = Exit Equity / Beginning Equity Check; IRR = (MOIC)^(1/n) - 1."
)

CACHED_WORKING_MODEL = None

KNOWN_ENTITIES = {
    "sbi": "SBIN",
    "state bank of india": "SBIN",
    "lic": "LICI",
    "lici": "LICI",
    "life insurance corporation": "LICI",
    "reliance": "RELIANCE",
    "tcs": "TCS",
    "tata consultancy": "TCS",
    "infosys": "INFY",
    "infy": "INFY",
    "hdfc": "HDFCBANK",
    "hdfc bank": "HDFCBANK",
    "icici": "ICICIBANK",
    "icici bank": "ICICIBANK",
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

def extract_pdf_document_text(uploaded_payload, query: str = "", max_chars: int = 25000) -> tuple[str, int, int]:
    """Reliable PDF extractor handling raw byte bundles, files, strings, or streams."""
    if uploaded_payload is None:
        return "", 0, 0

    if isinstance(uploaded_payload, dict) and "text" in uploaded_payload:
        txt = str(uploaded_payload["text"]).strip()
        return txt, 1, len(txt)

    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return "⚠️ [PDF Error]: 'pypdf' package is initializing in the runtime.", 0, 0

    try:
        raw_bytes = None
        if isinstance(uploaded_payload, dict):
            raw_bytes = uploaded_payload.get("data", b"")
        elif hasattr(uploaded_payload, "getvalue"):
            raw_bytes = uploaded_payload.getvalue()
        elif isinstance(uploaded_payload, bytes):
            raw_bytes = uploaded_payload
        elif hasattr(uploaded_payload, "read"):
            uploaded_payload.seek(0)
            raw_bytes = uploaded_payload.read()

        if not raw_bytes or len(raw_bytes) == 0:
            return "⚠️ [PDF Error]: Uploaded file contains 0 bytes.", 0, 0

        reader = PdfReader(io.BytesIO(raw_bytes))
        total_pages = len(reader.pages)
        if total_pages == 0:
            return "⚠️ [PDF Error]: Document has 0 pages.", 0, 0

        extracted_pages = []
        for idx in range(total_pages):
            try:
                page_str = reader.pages[idx].extract_text() or ""
                if page_str.strip():
                    extracted_pages.append((idx + 1, page_str.strip()))
            except Exception:
                continue

        if not extracted_pages:
            return "⚠️ [PDF Error]: Document contains no selectable text (scanned image).", total_pages, 0

        full_text = "\n\n".join([f"--- [PAGE {p_num}] ---\n{p_txt}" for p_num, p_txt in extracted_pages])
        return full_text, len(extracted_pages), len(full_text)

    except Exception as e:
        return f"⚠️ [PDF Parsing Error]: {str(e)}", 0, 0

def extract_and_fetch_live_data(prompt: str, status_callback=None) -> str:
    injected_context = []
    p_lower = prompt.lower()

    stock_keywords = ["stock", "share", "shares", "price", "market cap", "valuation", "52-week", "quote", "ticker", "trading at", "pe ratio"]
    has_dollar_ticker = bool(re.search(r'\$[A-Za-z]{2,10}\b', prompt))
    is_stock_request = any(k in p_lower for k in stock_keywords) or has_dollar_ticker

    target_tickers = []
    if is_stock_request:
        found_tickers = set()
        for name, sym in KNOWN_ENTITIES.items():
            if re.search(r'\b' + re.escape(name) + r'\b', p_lower):
                found_tickers.add(sym)
        for dt in re.findall(r'\$([A-Za-z]{2,10})\b', prompt):
            if dt.upper() not in FINANCIAL_ACRONYMS_BLACKLIST:
                found_tickers.add(dt.upper())
        for c in re.findall(r'\b[A-Z]{2,10}\b', prompt):
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
                market_rows.append(f"- {cname} ({resolved}): Current Price = {price}, Market Cap = {mcap}, 52-Week High = {high}, 52-Week Low = {low}")
            except Exception:
                continue
        if market_rows:
            injected_context.append("[VERIFIED LIVE EXCHANGE DATA]:\n" + "\n".join(market_rows))

    is_pure_stock_quote = bool(target_tickers) and is_stock_request
    if not is_pure_stock_quote:
        theory_bypass_terms = [
            "dcf", "discounted cash flow", "wacc", "waterfall", "restructuring",
            "alm", "nim", "deposit beta", "rate cut", "rate hike", "bps",
            "afs", "tce", "tangible common equity", "asset-sensitive", "fulcrum",
            "covenant", "debt capacity", "leverage", "coverage ratio", "accretion",
            "dilution", "equity value bridge", "treasury stock method", "net debt",
            "lbo", "moic", "irr", "sources and uses", "exit multiple", "credit agreement"
        ]
        is_pure_theory = any(w in p_lower for w in theory_bypass_terms)
        if not is_pure_theory:
            if status_callback:
                status_callback("info", "Autonomous research: retrieving verified records...")
            scraped_intel = dynamic_web_search(prompt)
            if scraped_intel:
                injected_context.append(f"[VERIFIED RESEARCH RECORDS]:\n{scraped_intel}")

    return "\n\n" + "\n\n".join(injected_context) + "\n\n" if injected_context else ""

def sanitize_report_formatting(text: str) -> str:
    """Cleans up markdown formatting, unescapes code wrappers, and guarantees complete returns."""
    if not text or not text.strip():
        return "No analysis generated. Please check your query or filing input."

    cleaned = re.sub(r'^```(?:markdown|text)?\s*\n', '', text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\n```\s*$', '', cleaned.strip())
    cleaned = re.sub(r'^\[(?:ORYN REPORT\vert{}REPORT\vert{}ANALYSIS\vert{}QUANTITATIVE ANALYSIS)\]:?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(?i)^(?:based on the (?:provided|available|attached) (?:information|data|filing|credit agreement)[,\s]*\n?)+', '', cleaned)
    cleaned = re.sub(r'\$\s*([0-9])', r'USD \1', cleaned)
    cleaned = re.sub(r'\|\s*\|', '|\n|', cleaned)

    if not cleaned.strip():
        return text.strip()

    return cleaned.strip()

def is_quantitative_model_request(prompt: str) -> bool:
    p = prompt.lower()
    dcf_terms = ["dcf", "discounted cash flow", "sensitivity matrix", "unlevered", "run a financial model", "build a financial model", "run a 5-year", "run a 10-year", "export the financial model to excel"]
    if any(t in p for t in dcf_terms):
        return True
    return bool(any(m in p for m in ["wacc", "discount rate", "terminal growth"]) and any(m in p for m in ["ebitda margin", "revenue growth", "cash flow projection"]))

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

    filter_out = ["whisper", "guard", "audio", "orpheus", "vision", "embed", "safeguard", "qwen"]
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
    
    # Target 1500 tokens for institutional depth; Zero out penalties to prevent token-suppression truncation
    kwargs["max_tokens"] = kwargs.get("max_tokens", 1500)
    kwargs["temperature"] = kwargs.get("temperature", 0.1)
    kwargs["frequency_penalty"] = 0.0
    kwargs["presence_penalty"] = 0.0
    kwargs["model"] = active_model

    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        err_str = str(e).lower()
        if any(term in err_str for term in ["output tokens per minute", "otpm", "reduce max_tokens", "rate_limit_exceeded"]):
            kwargs["max_tokens"] = 850
            alt_model = "llama-3.1-8b-instant" if active_model != "llama-3.1-8b-instant" else "llama-3.3-70b-versatile"
            kwargs["model"] = alt_model
            if status_callback:
                status_callback("info", f"Adjusting token quota (OTPM safeguard active)... retrying with {alt_model}.")
            return client.chat.completions.create(**kwargs)
        if any(term in err_str for term in ["404", "model_not_found", "decommissioned", "does not exist", "not have access"]):
            alt_model = get_best_available_model(client, exclude_model=active_model, force_refresh=True)
            kwargs["model"] = alt_model
            return client.chat.completions.create(**kwargs)
        if "429" in err_str:
            time.sleep(2.0)
            alt_model = get_best_available_model(client, exclude_model=active_model, force_refresh=True)
            kwargs["model"] = alt_model
            return client.chat.completions.create(**kwargs)
        raise e

def build_contextual_messages(user_input: str, system_prompt: str) -> list[dict]:
    history = []
    vault_context = []
    try:
        import streamlit as st
        # 1. Recover vault memories across resets
        if "session_vault" in st.session_state and isinstance(st.session_state.session_vault, list):
            for s_idx, session_turns in enumerate(st.session_state.session_vault[-3:]):
                summary_lines = []
                for m in session_turns:
                    role_lbl = "USER" if m.get("role") == "user" else "ANALYST"
                    content_snip = str(m.get("content", ""))[:400].replace("\n", " ")
                    summary_lines.append(f"{role_lbl}: {content_snip}")
                if summary_lines:
                    vault_context.append(f"[ARCHIVED RUN #{s_idx + 1}]:\n" + "\n".join(summary_lines))

        # 2. Recover active messages from current screen
        if "messages" in st.session_state and isinstance(st.session_state.messages, list):
            stored = st.session_state.messages
            candidates = stored[:-1] if (stored and stored[-1].get("role") == "user") else stored
            for m in candidates[-4:]:
                role = m.get("role", "user")
                content = str(m.get("content", "")).strip()
                if role in ["user", "assistant"] and content:
                    history.append({"role": role, "content": content})
    except Exception:
        pass

    prompt_with_vault = system_prompt
    if vault_context:
        prompt_with_vault += "\n\nPREVIOUS SESSIONS ARCHIVE (RETAINED FROM RESETS):\n" + "\n\n".join(vault_context)

    return [{"role": "system", "content": prompt_with_vault}] + history + [{"role": "user", "content": user_input}]

def parse_dcf_assumptions(prompt: str) -> dict:
    p_lower = prompt.lower()
    growth_match = re.search(r'(?:growing|growth rate|revenue growth|growth)[\s:]*(?:at\s*)?([0-9.]+)%', p_lower)
    growth = float(growth_match.group(1)) / 100.0 if growth_match else 0.20

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
        base_revenue = float(m_gen.group(1)) if m_gen else 50.0

    margin_match = re.search(r'(?:ebitda margin|margin|ebitda)[\s:]*(?:apparently\s*)?([0-9.]+)%', p_lower)
    margin = float(margin_match.group(1)) / 100.0 if margin_match else 0.25

    wacc_match = re.search(r'(?:wacc|discount rate|cost of capital)[\s:]*(?:apparently\s*)?([0-9.]+)%', p_lower)
    wacc = float(wacc_match.group(1)) / 100.0 if wacc_match else 0.10

    tv_match = re.search(r'(?:terminal growth|terminal rate|perpetual growth)[\s:]*(?:apparently\s*)?([0-9.]+)%', p_lower)
    tv_rate = float(tv_match.group(1)) / 100.0 if tv_match else 0.025

    return {"base_revenue": base_revenue, "growth": growth, "margin": margin, "wacc": wacc, "tv_rate": tv_rate}

def generate_deterministic_dcf_artifacts(user_input: str) -> tuple[str, bytes | None, bytes | None]:
    params = parse_dcf_assumptions(user_input)
    base_rev, growth, margin, wacc, tv_rate = params["base_revenue"], params["growth"], params["margin"], params["wacc"], params["tv_rate"]
    years = 5

    if tv_rate >= wacc:
        return f"⚠️ **DCF Model Exception**: Terminal growth ({tv_rate:.1%}) must be strictly less than WACC ({wacc:.1%}).", None, None

    years_idx = list(range(1, years + 1))
    rev = [base_rev * ((1 + growth) ** t) for t in years_idx]
    ebitda = [r * margin for r in rev]
    df = [1 / ((1 + wacc) ** t) for t in years_idx]
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
        f"* **Model Assumptions:** Year-1 Rev: USD {rev[0]:.2f}M • Growth: {growth:.1%} • Margin: {margin:.1%} • WACC: {wacc:.1%} • Terminal Rate: {tv_rate:.2%}\n"
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
    uploaded_pdf=None,
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

    # 1. Document Extraction & Active Grounding
    pdf_context = ""
    if uploaded_pdf is not None:
        if status_callback:
            status_callback("info", "Extracting institutional filing sections...")
        pdf_text, p_count, c_count = extract_pdf_document_text(uploaded_pdf, query=user_input)

        if pdf_text.startswith("⚠️"):
            return pdf_text, None, None, time.perf_counter() - start_time

        if c_count < 100:
            doc_name = uploaded_pdf.get("name", "Document") if isinstance(uploaded_pdf, dict) else "Document"
            err_msg = (
                f"⚠️ **Incomplete Filing Attached**: The attached file (`{doc_name}`) contains only {c_count} characters.\n"
                f"Please attach a PDF with selectable text or use the 1-click benchmark in the `➕` drawer."
            )
            return err_msg, None, None, time.perf_counter() - start_time

        if status_callback:
            status_callback("info", f"Extracted {p_count} page(s) ({c_count:,} characters) from document.")

        pdf_context = (
            f"[OFFICIAL FINANCIAL FILING DOCUMENT ATTACHED]:\n"
            f"\"\"\"\n{pdf_text}\n\"\"\"\n\n"
            f"CRITICAL DIRECTIVE: The user's query pertains directly to the document above. Perform all quantitative "
            f"calculations, covenant compliance tests, and balance sheet waterfalls DIRECTLY and FULLY using the "
            f"exact numbers, sections, and definitions in this filing. Answer all three parts systematically. Do not truncate.\n\n"
        )
    elif any(k in user_input.lower() for k in ["attached", "credit agreement", "indenture", "filing"]):
        return (
            "⚠️ **No Filing Attached**: You requested an analysis based on an attached document, "
            "but no document is currently loaded in this session. Click the **`➕` button** inside the typing bar "
            "to attach your PDF document, then re-send your query.",
            None, None, time.perf_counter() - start_time
        )

    # 2. Extract live market exchange data if requested
    context_data = extract_and_fetch_live_data(user_input, status_callback=status_callback)

    augmented_prompt = f"{pdf_context}USER DIRECTIVE:\n{user_input}\n\n{context_data}".strip()

    try:
        messages = build_contextual_messages(augmented_prompt, SYSTEM_PROMPT)
        resp = create_completion_with_fallback(
            client=client,
            preferred_model=selected_model,
            status_callback=status_callback,
            messages=messages,
            max_tokens=1500
        )
        final_text = resp.choices[0].message.content or ""
        elapsed = time.perf_counter() - start_time
        return sanitize_report_formatting(final_text), None, None, elapsed
    except Exception as e:
        return f"Error: {e}", None, None, time.perf_counter() - start_time
