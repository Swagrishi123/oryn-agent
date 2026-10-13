import json
import re
import requests
import yfinance as yf

# Tool schemas registered for Groq
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_financial_data",
            "description": "Fetch real-time stock price, 52-week range, and market cap for Indian (NSE/BSE) or global equities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol, e.g., RELIANCE, TCS, INFY, HDFCBANK, AAPL"
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]

def fetch_from_google_finance(symbol: str) -> dict | None:
    """Scrapes Google Finance with browser headers as a reliable fallback."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    for exchange in ["NSE", "BOM"]:
        url = f"https://www.google.com/finance/quote/{symbol}:{exchange}"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                html = res.text
                price_match = re.search(r'class="YMlKec fxKbKc">₹?([0-9,.]+)<', html)
                mcap_match = re.search(r'Market cap</div>\s*<div[^>]*>([^<]+)<', html)
                pe_match = re.search(r'P/E ratio</div>\s*<div[^>]*>([^<]+)<', html)
                high_match = re.search(r'52-wk high</div>\s*<div[^>]*>₹?([^<]+)<', html)
                low_match = re.search(r'52-wk low</div>\s*<div[^>]*>₹?([^<]+)<', html)

                if price_match:
                    price_val = float(price_match.group(1).replace(",", "").strip())
                    return {
                        "company_name": symbol,
                        "ticker_resolved": f"{symbol}.{exchange}",
                        "current_price": f"₹{price_val:,.2f}",
                        "market_cap": mcap_match.group(1).strip() if mcap_match else "N/A",
                        "pe_ratio": pe_match.group(1).strip() if pe_match else "N/A",
                        "52w_high": f"₹{high_match.group(1).strip()}" if high_match else "N/A",
                        "52w_low": f"₹{low_match.group(1).strip()}" if low_match else "N/A",
                        "currency": "INR"
                    }
        except Exception:
            continue
    return None

def tool_get_financial_data(ticker: str) -> dict:
    """Retrieves live market data using direct Yahoo v8 chart endpoint with Google Finance fallback."""
    clean_sym = ticker.strip().upper().replace("$", "")
    
    candidates = []
    if not clean_sym.endswith((".NS", ".BO")):
        candidates.append(f"{clean_sym}.NS")
        candidates.append(f"{clean_sym}.BO")
    candidates.append(clean_sym)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    # Strategy 1: Direct Yahoo Finance v8 chart query
    for sym in candidates:
        try:
            chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=1d"
            res = requests.get(chart_url, headers=headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                if meta and ("regularMarketPrice" in meta or "chartPreviousClose" in meta):
                    price = float(meta.get("regularMarketPrice") or meta.get("chartPreviousClose") or 0.0)
                    high_52 = meta.get("fiftyTwoWeekHigh", "N/A")
                    low_52 = meta.get("fiftyTwoWeekLow", "N/A")
                    curr = meta.get("currency", "INR")
                    
                    # Fetch Market Cap & PE via Google Finance or fast_info
                    gf_data = fetch_from_google_finance(clean_sym)
                    mcap = gf_data.get("market_cap") if gf_data else "N/A"
                    pe = gf_data.get("pe_ratio") if gf_data else "N/A"

                    is_inr = curr == "INR" or ".NS" in sym or ".BO" in sym
                    price_fmt = f"₹{price:,.2f}" if is_inr else f"${price:,.2f}"
                    high_fmt = f"₹{high_52:,.2f}" if is_inr and isinstance(high_52, (int, float)) else str(high_52)
                    low_fmt = f"₹{low_52:,.2f}" if is_inr and isinstance(low_52, (int, float)) else str(low_52)

                    return {
                        "company_name": clean_sym,
                        "ticker_resolved": sym,
                        "current_price": price_fmt,
                        "market_cap": mcap,
                        "pe_ratio": pe,
                        "52w_high": high_fmt,
                        "52w_low": low_fmt,
                        "currency": curr
                    }
        except Exception:
            continue

    # Strategy 2: Google Finance direct fallback
    gf_fallback = fetch_from_google_finance(clean_sym)
    if gf_fallback:
        return gf_fallback

    return {"error": f"Connection to NSE/BSE feeds timed out for '{ticker}'."}

def tool_web_search(query: str) -> str:
    return f"Live search results for: '{query}'."
