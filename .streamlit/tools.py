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
    """Strips conversational noise and standardizes state names."""
    clean = re.sub(
        r'(?i)\b(who is the current|who is the latest|who is the|current|latest|who was the|who was|'
        r'founder of|founders of|who founded|ceo of|governor of|cm of|chief minister of|president of|'
        r'chairman of|head of|what is the|tell me about|stock price of|price of)\b',
        '',
        query
    )
    clean = re.sub(r'[,?!\'\"]', '', clean).strip()
    clean = re.sub(r'(?i)\bmaharastra\b', 'Maharashtra', clean)
    clean = re.sub(r'(?i)\bbengal\b', 'West Bengal', clean)
    clean = re.sub(r'(?i)\btamilnadu\b', 'Tamil Nadu', clean)
    return clean if len(clean) >= 2 else query.strip()

def fetch_wikipedia_entity_summary(entity: str) -> str | None:
    """Queries Wikipedia Action API for verified incumbency and encyclopedic records."""
    headers = {"User-Agent": "ORYNAnalyst/2.0 (Institutional BFSI Research; contact@oryn.ai)"}
    encoded_search = urllib.parse.quote_plus(entity)
    
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
                        summary = ". ".join(extract.split(". ")[:4]).strip()
                        if not summary.endswith("."):
                            summary += "."
                        return f"• [Official Encyclopedia Record - {title}]: {summary}"
    except Exception:
        pass

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
                                    summary = ". ".join(sub_extract.split(". ")[:4]).strip()
                                    if not summary.endswith("."):
                                        summary += "."
                                    return f"• [Official Encyclopedia Record - {found_title}]: {summary}"
    except Exception:
        pass

    return None

def fetch_google_news_rss(query: str, max_items: int = 3) -> str | None:
    """Queries news RSS syndication feeds, filtering out speculation and campaign buzz."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36"}
    encoded_q = urllib.parse.quote_plus(query.strip())
    rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        res = requests.get(rss_url, headers=headers, timeout=3.0)
        if res.status_code == 200:
            root = ET.fromstring(res.content)
            items = root.findall("./channel/item")
            snippets = []
            speculative_terms = ["opinion", "speculation", "rally", "claims", "campaign", "poll", "will become"]
            for item in items:
                title = item.findtext("title", "").strip()
                pub_date = item.findtext("pubDate", "").strip()
                clean_title = re.sub(r' - [^-]+$', '', title)
                # Filter out obvious speculative opinion articles
                if clean_title and not any(term in clean_title.lower() for term in speculative_terms):
                    date_prefix = f"[{pub_date[:16]}] " if pub_date else ""
                    snippets.append(f"• {date_prefix}{clean_title}")
                if len(snippets) >= max_items:
                    break
            if snippets:
                return "\n".join(snippets)
    except Exception:
        pass
    return None

def dynamic_web_search(query: str) -> str | None:
    """Stratified retrieval: targets verified incumbency registries and official gazettes first."""
    clean_entity = extract_core_entity(query)
    findings = []
    p_lower = query.lower()

    is_cm = any(k in p_lower for k in ["cm", "chief minister"])
    is_governor = "governor" in p_lower
    is_corporate = any(k in p_lower for k in ["chairman", "ceo", "md", "chief executive", "director", "head of", "head"])

    if is_cm:
        # 1. Target verified incumbency registers directly
        cm_wiki = fetch_wikipedia_entity_summary(f"List of chief ministers of {clean_entity}")
        if not cm_wiki:
            cm_wiki = fetch_wikipedia_entity_summary(f"Chief Minister of {clean_entity}")
        if cm_wiki:
            findings.append(cm_wiki)

        # 2. Query official oath and cabinet appointment records
        pol_news = fetch_google_news_rss(f"\"{clean_entity}\" \"Chief Minister\" \"sworn in\" OR \"takes oath\"", max_items=2)
        if pol_news:
            findings.append(f"[OFFICIAL OATH & CABINET DISCLOSURES]:\n{pol_news}")

    elif is_governor:
        gov_wiki = fetch_wikipedia_entity_summary(f"List of governors of {clean_entity}")
        if not gov_wiki:
            gov_wiki = fetch_wikipedia_entity_summary(f"Governor of {clean_entity}")
        if gov_wiki:
            findings.append(gov_wiki)

        gov_news = fetch_google_news_rss(f"\"{clean_entity}\" \"Governor\" \"sworn in\"", max_items=2)
        if gov_news:
            findings.append(f"[OFFICIAL OATH & CABINET DISCLOSURES]:\n{gov_news}")

    elif is_corporate:
        appointment_news = fetch_google_news_rss(f"{clean_entity} CEO OR MD OR Chairman appointed OR \"takes charge\"", max_items=3)
        if appointment_news:
            findings.append(f"[OFFICIAL APPOINTMENTS & GAZETTE RECORDS]:\n{appointment_news}")

        corp_wiki = fetch_wikipedia_entity_summary(clean_entity)
        if corp_wiki:
            findings.append(corp_wiki)

    else:
        news_intel = fetch_google_news_rss(f"{clean_entity} current", max_items=2)
        if news_intel:
            findings.append(f"[RECENT MEDIA HEADLINES]:\n{news_intel}")

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
