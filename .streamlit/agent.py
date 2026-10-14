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
    "USD", "INR", "NSE", "BSE", "FOUNDER", "FOUNDERS", "OF", "TSM", "MOIC",
    "CET1", "RWA", "LCR", "NSFR", "HQLA"
}

SYSTEM_PROMPT = (
    "You are ORYN Analyst, an elite institutional BFSI quantitative analyst and corporate intelligence advisor.\n\n"
    "CRITICAL CONVERSATION & MEMORY PROTOCOL:\n"
    "1. MULTI-TURN CONTINUITY: Always review previous turns in the conversation. When a user asks to 're-run', "
    "'sensitize', or 'modify' a previous calculation, retain ALL baseline parameters (EBITDA, purchase price, equity check, "
    "debt balances) from the preceding prompt and ONLY update the requested variable.\n"
    "2. NO BRACKET TAGS OR ECHOING: NEVER output bracketed headers (e.g., [ORYN REPORT]). Begin directly with the analysis.\n"
    "3. NO CHATBOT HEDGING: Eliminate conversational filler (e.g., 'Based on the available information'). Deliver crisp intelligence.\n"
    "4. REAL-WORLD INCUMBENCY:\n"
    "   - Tamil Nadu CM: M. K. Stalin (DMK), in office since May 2021.\n"
    "   - Maharashtra CM: Devendra Fadnavis (BJP / Mahayuti), in office since December 2024.\n"
    "   - SBI Chairman: Challa Sreenivasulu Setty (C.S. Setty), appointed August 2024.\n"
    "   - LIC CEO & MD: R. Doraiswamy, appointed July 2025.\n"
    "   - SpaceX: Founded in 2002 by Elon Musk.\n\n"
    "FIG BANKING & CORPORATE FINANCE LOGIC (STRICT ADHERENCE):\n"
    "1. COMMERCIAL BANK COVENANTS & LEVERAGE:\n"
    "   - Regulated commercial banks DO NOT have 'Debt-to-EBITDA' covenants. Deposits and debt are funding liabilities, not corporate debt.\n"
    "   - Banks are governed by Basel III Regulatory Capital Requirements: CET1 ratio (>= 4.5%), Tier 1 ratio (>= 6.0%), "
    "Total Capital ratio (>= 8.0%), and the Basel III Leverage Ratio (Tier 1 Capital / Total Leverage Exposure >= 3.0%).\n"
    "   - When answering general/conceptual questions without provided financial statements, provide the institutional regulatory "
    "framework rather than inventing fictional case studies or numbers.\n"
    "2. ACCOUNTING DEFINITIONS:\n"
    "   - EBITDA is Earnings BEFORE Interest, Taxes, Depreciation, and Amortization. Interest expense NEVER alters or reduces EBITDA.\n"
    "3. VALUATION & RESTRUCTURING WATERFALLS:\n"
    "   - Equity Value = Implied EV - Net Debt - Minority Interest - Pension Deficit + Equity Investments in Affiliates.\n"
    "   - Net Debt = Total Gross Debt - Unrestricted Cash (exclude restricted cash).\n"
    "   - Chapter 11 priority: DIP -> First Lien -> Second Lien -> Senior Unsecured -> Subordinated. Fulcrum security is partially impaired.\n"
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

def extract_and_fetch_live_data(prompt: str, status_callback=None) -> str:
    injected_context = []
    p_lower = prompt.lower()

    stock_keywords = [
        "stock", "share", "shares", "price", "market cap", "valuation",
        "52-week", "quote", "ticker", "trading at", "pe ratio", "dividend",
        "rupees", "inr", "usd", "market price"
    ]
    has_dollar_ticker = bool(re.search(r'\$[A-Za-z]{2,10}\b', prompt))
    is_stock_request = any(k in p_lower for k in stock_keywords) or has_dollar_ticker

    target_tickers = []
    if is_stock_request:
        found_tickers = set()
        for name, sym in KNOWN_ENTITIES.items():
            if re.search(r'\b' + re.escape(name) + r'\b', p_lower):
                found_tickers.add(sym)

        dollar_tickers = re.findall(r'\$([A-Za-z]{2,10})\b', prompt)
        for dt in dollar_tickers:
            dt_clean = dt.upper()
            if dt_clean not in FINANCIAL_ACRONYMS_BLACKLIST:
                found_tickers.add(dt_clean)

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
                "[VERIFIED LIVE EXCHANGE DATA]:\n" + "\n".join(market_rows)
            )

    is_pure_stock_quote = bool(target_tickers) and is_stock_request
    if not is_pure_stock_quote:
        theory_bypass_terms = [
            "dcf", "discounted cash flow", "wacc", "waterfall", "restructuring",
            "alm", "nim", "deposit beta", "rate cut", "rate hike", "bps",
            "afs", "tce", "tangible common equity", "asset-sensitive", "fulcrum",
            "covenant", "debt capacity", "leverage", "coverage ratio", "accretion",
            "dilution", "equity value bridge", "treasury stock method", "net debt",
            "lbo", "moic", "irr", "sources and uses", "exit multiple"
        ]
        is_pure_theory = any(w in p_lower for w in theory_bypass_terms)
        
        if not is_pure_theory:
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
    """Repairs tables, guarantees newlines between table rows, and purges leaked tags."""
    cleaned = re.sub(r'\[[A-Za-z0-9\s\-_\.]+\]:?\s*', '', text)
    cleaned = re.sub(r'```(?:python)?\s*[\s\S]*?```', '', cleaned)
    
    filler_patterns = [
        r'(?i)^please provide the requested information[^\n]*\n?',
        r'(?i)based on the (?:available )?(?:information|data)[,\s]*',
        r'(?i)please note that (?:the )?(?:information|data)[,\s\S]*?(?=\n|$)',
        r'(?i)for more detailed information[,\s\S]*?(?=\n|$)'
    ]
    for fp in filler_patterns:
        cleaned = re.sub(fp, '', cleaned)

    cleaned = re.sub(r'\|\s*\|', '|\n|', cleaned)

    lines = cleaned.split('\n')
    output = []
    current_table = []

    def flush_tbl(tbl):
        if not tbl:
            return []
        if len(tbl) >= 2 and not any('---' in r for r in tbl[:2]):
            header_cols = [c.strip() for c in tbl[0].strip('|').split('|')]
            divider = "| " + " | ".join([":---"] * len(header_cols)) + " |
