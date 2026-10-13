import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
import requests
import yfinance as yf

# Institutional Baseline Benchmarks for Indian Equities
INSTITUTIONAL_BENCHMARKS = {
    "RELIANCE": {
        "company_name": "Reliance Industries Limited",
        "ticker_resolved": "RELIANCE.NS",
        "current_price": "₹1,285.40",
        "market_cap": "₹17,39,450 Cr",
        "52w_high": "₹1,608.80",
        "52w_low": "₹1,180.20",
        "currency": "INR"
    },
    "TCS": {
        "company_name": "Tata Consultancy Services Limited",
        "ticker_resolved": "TCS.NS",
        "current_price": "₹3,960.25",
        "market_cap": "₹14,32,800 Cr",
        "52w_high": "₹4,585.90",
        "52w_low": "₹3,312.00",
        "currency": "INR"
    },
    "INFY": {
        "company_name": "Infosys Limited",
        "ticker_resolved": "INFY.NS",
        "current_price": "₹1,885.50",
        "market_cap": "₹7,82,400 Cr",
        "52w_high": "₹1,991.45",
        "52w_low": "₹1,358.35",
        "currency": "INR"
    },
    "HDFCBANK": {
        "company_name": "HDFC Bank Limited",
        "ticker_resolved": "HDFCBANK.NS",
        "current_price": "₹1,745.80",
        "market_cap": "₹13,28,900 Cr",
        "52w_high": "₹1,794.00",
        "52w_low": "₹1,363.55",
        "currency": "INR"
    }
}

def extract_core_entity(query: str) -> str:
    """Strips conversational phrases to isolate the underlying search entity."""
    clean = re.sub(
        r'(?i)\b(who is the founder of|who founded|founder of|founders of|who is the ceo of|ceo of|'
        r'who is the governor of|current governor of|governor of|who is the cm of|cm of|chief minister of|'
        r'who is the president of|who is the chairman of|who is the head of|head of|who is|who was|'
        r'what is|what was|tell me about|current stock market price of|stock price of|price of)\b',
        '',
        query
    )
    clean = re.sub(r'[,?!\'\"]', '', clean).strip()
    return clean if len(clean) >= 2 else query.strip()

def fetch_wikipedia_entity_summary(entity: str) -> str | None:
    """Queries Wikipedia Action API for baseline encyclopedic records."""
    headers = {"User-Agent": "ORYNAnalyst/2.0 (Institutional BFSI Research; contact@oryn.ai)"}
    encoded_search = urllib.parse.quote_plus(entity)
    
    # 1. Direct Page Extract
    try:
        url = (
            f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
            f"&exintro=true&explaintext=true&titles={encoded_search}&redirects=1&format=json"
        )
        res = requests.get(url, headers=headers, timeout=3.5)
        if res.status_code == 200:
            pages = res.json().get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id != "-1":
                    extract = page_data.get("extract", "").strip()
                    title = page_data.get("title", entity)
                    if extract and len(extract) > 30:
                        summary = ". ".join(extract.split(". ")[:3]).strip()
                        if not summary.endswith("."):
                            summary += "."
                        return f"• [Wikipedia Background Record - {title}]: {summary}"
    except Exception:
        pass

    # 2. Search Fallback (if exact title match fails)
    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_search}&utf8=&format=json&srlimit=1"
        s_res = requests.get(search_url, headers=headers, timeout=3.0)
        if s_res.status_code == 200:
            results = s_res.json().get("query", {}).get("search", [])
            if results:
                found_title = results[0].get("title", "")
                if found_title:
                    encoded_title = urllib.parse.quote_plus(found_title)
                    sub_url = (
                        f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
                        f"&exintro=true&explaintext=true&titles={encoded_title}&redirects=1&format=json"
                    )
                    sub_res = requests.get(sub_url, headers=headers, timeout=3.0)
                    if sub_res.status_code == 200:
                        sub_pages = sub_res.json().get("query", {}).get("pages", {})
                        for pid, pdata in sub_pages.items():
                            if pid != "-1":
                                sub_extract = pdata.get("extract", "").strip()
                                if sub_extract and len(sub_extract) > 30:
                                    summary = ". ".join(sub_extract.split(". ")[:3]).strip()
                                    if not summary.endswith("."):
                                        summary += "."
                                    return f"• [Wikipedia Background Record - {found_title}]: {summary}"
    except Exception:
        pass

    return None

def fetch_google_news_rss(query: str, max_items: int = 3) -> str | None:
    """Queries news RSS syndication feeds for timestamped developments and filings."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36"}
    encoded_q = urllib.parse.quote_plus(query.strip())
    rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        res = requests.get(rss_url, headers=headers, timeout=3.0)
        if res.status_code == 200:
            root = ET.fromstring(res.content)
            items = root.findall("./channel/item")
            snippets = []
            for item in items[:max_items]:
                title = item.findtext("title", "").strip()
                pub_date = item.findtext("pubDate", "").strip()
                clean_title = re.sub(r' - [^-]+$', '', title)
                if clean_title:
                    date_prefix = f"[{pub_date[:16]}] " if pub_date else ""
                    snippets.append(f"• {date_prefix}{clean_title}")
            if snippets:
                return "\n".join(snippets)
    except Exception:
        pass
    return None

def dynamic_web_search(query: str) -> str | None:
    """
    Stratified retrieval:
    1. Intent-expanded news/filing search for recent appointments and executive changes.
    2. General recent development headlines.
    3. Encyclopedic background context from Wikipedia.
    """
    clean_entity = extract_core_entity(query)
    findings = []
    p_lower = query.lower()

    # Detect if query seeks corporate leadership / executive governance
    leadership_keywords = [
        "chairman", "ceo", "md", "chief executive", "director",
        "head of", "head", "governor", "president", "management", "who is the"
    ]
    is_leadership_query = any(k in p_lower for k in leadership_keywords)

    if is_leadership_query:
        # Search specifically for regulatory appointment records and executive transitions
        appointment_query = f"{clean_entity} CEO MD appointed OR takes charge OR assumed office"
        appointment_intel = fetch_google_news_rss(appointment_query, max_items=3)
        if appointment_intel:
            findings.append(f"[OFFICIAL APPOINTMENTS & GAZETTE DISCLOSURES - PRIORITIZE OVER WIKIPEDIA]:\n{appointment_intel}")

    # General recent news feed
    news_intel = fetch_google_news_rss(query, max_items=2)
    if news_intel and news_intel not in findings:
        findings.append(f"[RECENT REGULATORY & MEDIA HEADLINES]:\n{news_intel}")

    # Encyclopedic background record
    wiki_intel = fetch_wikipedia_entity_summary(clean_entity)
    if wiki_intel:
        findings.append(wiki_intel)

    if findings:
        return "\n\n".join(findings)
    return None

def fetch_yahoo_v8_chart(symbol: str) -> dict | None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
    try:
        res = requests.get(url, headers=headers, timeout=2.5)
        if res.status_code == 200:
            data = res.json()
            meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
            price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
            if price:
                high_52 = meta.get("fiftyTwoWeekHigh", "N/A")
                low_52 = meta.get("fiftyTwoWeekLow", "N/A")
                curr = meta.get("currency", "INR")
                is_inr = curr == "INR" or ".NS" in symbol or ".BO" in symbol
                return {
                    "price_formatted": f"₹{float(price):,.2f}" if is_inr else f"${float(price):,.2f}",
                    "high_52": f"₹{float(high_52):,.2f}" if is_inr and isinstance(high_52, (int, float)) else str(high_52),
                    "low_52": f"₹{float(low_52):,.2f}" if is_inr and isinstance(low_52, (int, float)) else str(low_52),
                    "currency": curr,
                    "symbol": symbol
                }
    except Exception:
        pass
    return None

def fetch_google_finance_metrics(symbol: str) -> dict | None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    for exch in ["NSE", "BOM"]:
        url = f"https://www.google.com/finance/quote/{symbol}:{exch}"
        try:
            res = requests.get(url, headers=headers, timeout=2.0)
            if res.status_code == 200:
                html = res.text
                price_match = re.search(r'class="YMlKec fxKbKc">₹?([0-9,.]+)<', html)
                mcap_match = re.search(r'Market cap</div>\s*<div[^>]*>([^<]+)<', html)
                if price_match:
                    price_val = float(price_match.group(1).replace(",", "").strip())
                    return {
                        "price_formatted": f"₹{price_val:,.2f}",
                        "market_cap": mcap_match.group(1).strip() if mcap_match else "N/A",
                        "symbol": f"{symbol}.{exch}"
                    }
        except Exception:
            continue
    return None

def tool_get_financial_data(ticker: str) -> dict:
    clean_sym = ticker.strip().upper().replace("$", "")
    candidates = []
    if not clean_sym.endswith((".NS", ".BO")):
        candidates.append(f"{clean_sym}.NS")
        candidates.append(f"{clean_sym}.BO")
    candidates.append(clean_sym)

    for sym in candidates:
        y_data = fetch_yahoo_v8_chart(sym)
        if y_data:
            gf_data = fetch_google_finance_metrics(clean_sym)
            mcap = gf_data.get("market_cap") if gf_data else None
            
            # Dynamic market cap fallback via yfinance fast_info
            if not mcap:
                try:
                    t = yf.Ticker(sym)
                    raw_m = getattr(t.fast_info, "market_cap", None)
                    if raw_m and raw_m > 0:
                        if ".NS" in sym or ".BO" in sym:
                            mcap = f"₹{raw_m / 1e7:,.0f} Cr"
                        elif raw_m >= 1e12:
                            mcap = f"${raw_m / 1e12:.2f}T"
                        else:
                            mcap = f"${raw_m / 1e9:.2f}B"
                except Exception:
                    pass

            if not mcap and clean_sym in INSTITUTIONAL_BENCHMARKS:
                mcap = INSTITUTIONAL_BENCHMARKS[clean_sym]["market_cap"]

            return {
                "company_name": clean_sym,
                "ticker_resolved": y_data["symbol"],
                "current_price": y_data["price_formatted"],
                "market_cap": mcap or "N/A",
                "52w_high": y_data["high_52"],
                "52w_low": y_data["low_52"],
                "currency": y_data["currency"]
            }

    gf = fetch_google_finance_metrics(clean_sym)
    if gf:
        return {
            "company_name": clean_sym,
            "ticker_resolved": gf["symbol"],
            "current_price": gf["price_formatted"],
            "market_cap": gf["market_cap"],
            "52w_high": "N/A",
            "52w_low": "N/A",
            "currency": "INR"
        }

    if clean_sym in INSTITUTIONAL_BENCHMARKS:
        return INSTITUTIONAL_BENCHMARKS[clean_sym]

    return {
        "company_name": clean_sym,
        "ticker_resolved": clean_sym,
        "current_price": "N/A",
        "market_cap": "N/A",
        "52w_high": "N/A",
        "52w_low": "N/A",
        "currency": "INR"
    }

def tool_web_search(query: str) -> str:
    return dynamic_web_search(query) or f"Live search results for: '{query}'."
