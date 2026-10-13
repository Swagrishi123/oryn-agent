import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import requests
import yfinance as yf

# No hardcoded prices are used as fallbacks: unavailable data must remain unavailable.
INSTITUTIONAL_BENCHMARKS = {}

def extract_core_entity(query: str) -> str:
    """Strips conversational phrases to isolate the search entity."""
    clean = re.sub(
        r'(?i)\b(who is the founder of|who founded|founder of|founders of|who is the ceo of|ceo of|'
        r'who is the governor of|current governor of|governor of|who is the cm of|cm of|chief minister of|'
        r'who is the president of|who is the chairman of|who is|who was|what is|what was|tell me about)\b',
        '',
        query
    )
    clean = re.sub(r'[,?!\'\"]', '', clean).strip()
    return clean if len(clean) >= 2 else query.strip()

def fetch_wikipedia_entity_summary(entity: str) -> str | None:
    """Queries Wikipedia REST summary API for factual encyclopedic data."""
    headers = {"User-Agent": "ORYNAnalyst/2.0 (Institutional BFSI Research)"}
    encoded_search = urllib.parse.quote_plus(entity)
    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_search}&utf8=&format=json&srlimit=1"
        res = requests.get(search_url, headers=headers, timeout=2.5)
        if res.status_code == 200:
            results = res.json().get("query", {}).get("search", [])
            if results:
                title = results[0].get("title", "")
                if title:
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                    s_res = requests.get(summary_url, headers=headers, timeout=2.5)
                    if s_res.status_code == 200:
                        extract = s_res.json().get("extract", "")
                        if extract and len(extract) > 40:
                            return f"• [Wikipedia result: {title}; source: Wikipedia]: {extract}"
    except Exception:
        pass
    return None

def fetch_google_news_rss(query: str, max_items: int = 3) -> str | None:
    """Queries machine-readable RSS syndication feeds without datacenter IP blocks."""
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
    """Layered retrieval: Encyclopedic entity lookup + timestamped news headlines."""
    clean_entity = extract_core_entity(query)
    findings = []
    wiki_intel = fetch_wikipedia_entity_summary(clean_entity)
    if wiki_intel:
        findings.append(wiki_intel)
    news_intel = fetch_google_news_rss(query)
    if news_intel:
        findings.append(news_intel)
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
            price = meta.get("regularMarketPrice")
            if price:
                high_52 = meta.get("fiftyTwoWeekHigh", "N/A")
                low_52 = meta.get("fiftyTwoWeekLow", "N/A")
                curr = meta.get("currency", "INR")
                is_inr = curr == "INR"
                return {
                    "price_formatted": f"₹{float(price):,.2f}" if is_inr else f"${float(price):,.2f}",
                    "high_52": f"₹{float(high_52):,.2f}" if is_inr and isinstance(high_52, (int, float)) else str(high_52),
                    "low_52": f"₹{float(low_52):,.2f}" if is_inr and isinstance(low_52, (int, float)) else str(low_52),
                    "currency": curr,
                    "symbol": symbol,
                    "company_name": meta.get("longName") or meta.get("shortName") or symbol,
                    "source": "Yahoo Finance chart endpoint",
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat()
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
                        "symbol": f"{symbol}.{exch}",
                        "company_name": symbol,
                        "source": "Google Finance page",
                        "retrieved_at_utc": datetime.now(timezone.utc).isoformat()
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
            base_symbol = clean_sym.removesuffix(".NS").removesuffix(".BO")
            gf_data = fetch_google_finance_metrics(base_symbol) if y_data.get("currency") == "INR" else None
            mcap = gf_data.get("market_cap") if gf_data else None

            return {
                "company_name": y_data.get("company_name", clean_sym),
                "ticker_resolved": y_data["symbol"],
                "current_price": y_data["price_formatted"],
                "market_cap": mcap or "N/A",
                "52w_high": y_data["high_52"],
                "52w_low": y_data["low_52"],
                "currency": y_data["currency"],
                "source": y_data.get("source", "Yahoo Finance chart endpoint"),
                "retrieved_at_utc": y_data.get("retrieved_at_utc")
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
            "currency": "INR",
            "source": gf.get("source", "Google Finance page"),
            "retrieved_at_utc": gf.get("retrieved_at_utc")
        }

    return {
        "company_name": clean_sym,
        "ticker_resolved": clean_sym,
        "current_price": "N/A",
        "market_cap": "N/A",
        "52w_high": "N/A",
        "52w_low": "N/A",
        "currency": "UNKNOWN",
        "source": None,
        "retrieved_at_utc": None,
        "data_status": "unavailable"
    }

def tool_web_search(query: str) -> str:
    return dynamic_web_search(query) or "No search results were retrieved. Do not infer that a live search succeeded."
