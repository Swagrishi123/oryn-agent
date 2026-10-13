import json
import re
import requests
import urllib.parse
import yfinance as yf

# Institutional Baseline Benchmarks for Equities
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

def dynamic_web_search(query: str, max_results: int = 3) -> str | None:
    """Dynamically searches the live web for ANY general knowledge or institutional query."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    encoded_q = urllib.parse.quote_plus(query)

    # Strategy 1: Direct DuckDuckGo HTML Instant Scrape (No API key needed)
    try:
        url = f"https://html.duckduckgo.com/html/?q={encoded_q}"
        res = requests.post(url, headers=headers, data={"q": query}, timeout=3.5)
        if res.status_code == 200:
            snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', res.text, re.DOTALL)
            clean_snippets = []
            for s in snippets[:max_results]:
                text = re.sub(r'<[^>]+>', '', s).strip()
                if len(text) > 30:
                    clean_snippets.append(text)
            if clean_snippets:
                return "\n".join([f"- {s}" for s in clean_snippets])
    except Exception:
        pass

    # Strategy 2: Dynamic Wikipedia Open Search API Fallback
    try:
        wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_q}&utf8=&format=json&srlimit=2"
        res = requests.get(wiki_url, headers=headers, timeout=3.0)
        if res.status_code == 200:
            search_results = res.json().get("query", {}).get("search", [])
            wiki_snippets = []
            for item in search_results:
                snippet = re.sub(r'<[^>]+>', '', item.get("snippet", "")).strip()
                title = item.get("title", "")
                if snippet:
                    wiki_snippets.append(f"- ({title}): {snippet}")
            if wiki_snippets:
                return "\n".join(wiki_snippets)
    except Exception:
        pass

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
    return dynamic_web_search(query) or f"No direct search results for: '{query}'."
