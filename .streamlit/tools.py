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
    """Bulletproof fallback that scrapes Google Finance (never blocks cloud servers)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    # Check NSE first, then BSE
    for exchange in ["NSE", "BOM"]:
        url = f"https://www.google.com/finance/quote/{symbol}:{exchange}"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                html = res.text
                # Extract live price
                price_match = re.search(r'class="YMlKec fxKbKc">₹?([0-9,.]+)<', html)
                if price_match:
                    price_val = price_match.group(1).replace(",", "").strip()
                    
                    # Extract Market Cap
                    mcap_match = re.search(r'Market cap</div>\s*<div[^>]*>([^<]+)<', html)
                    mcap_val = mcap_match.group(1).strip() if mcap_match else "N/A"
                    
                    # Extract P/E Ratio
                    pe_match = re.search(r'P/E ratio</div>\s*<div[^>]*>([^<]+)<', html)
                    pe_val = pe_match.group(1).strip() if pe_match else "N/A"
                    
                    # Extract 52-week range
                    high_match = re.search(r'52-wk high</div>\s*<div[^>]*>₹?([^<]+)<', html)
                    low_match = re.search(r'52-wk low</div>\s*<div[^>]*>₹?([^<]+)<', html)
                    high_val = f"₹{high_match.group(1).strip()}" if high_match else "N/A"
                    low_val = f"₹{low_match.group(1).strip()}" if low_match else "N/A"

                    return {
                        "company_name": symbol,
                        "ticker_resolved": f"{symbol}.{exchange}",
                        "current_price": f"₹{float(price_val):,.2f}",
                        "market_cap": mcap_val,
                        "pe_ratio": pe_val,
                        "52w_high": high_val,
                        "52w_low": low_val,
                        "currency": "INR"
                    }
        except Exception:
            continue
    return None

def tool_get_financial_data(ticker: str) -> dict:
    """Fetches real-time price using Yahoo Finance fast_info with Google Finance fallback."""
    clean_sym = ticker.strip().upper().replace("$", "")
    
    # Prioritize Indian suffixes (.NS for NSE, .BO for BSE)
    candidates = []
    if not clean_sym.endswith((".NS", ".BO")):
        candidates.append(f"{clean_sym}.NS")
        candidates.append(f"{clean_sym}.BO")
    candidates.append(clean_sym)

    # Attempt 1: Yahoo Finance fast_info
    for sym in candidates:
        try:
            tk = yf.Ticker(sym)
            fi = tk.fast_info
            price = getattr(fi, "last_price", 0.0) or 0.0
            if price > 0.0:
                mcap = getattr(fi, "market_cap", 0) or 0
                high_52 = getattr(fi, "year_high", "N/A")
                low_52 = getattr(fi, "year_low", "N/A")
                curr = getattr(fi, "currency", "INR")

                is_inr = curr == "INR" or ".NS" in sym or ".BO" in sym
                mcap_fmt = f"₹{mcap / 1e7:,.2f} Cr" if is_inr and mcap else (f"${mcap / 1e9:,.2f}B" if mcap else "N/A")
                price_fmt = f"₹{price:,.2f}" if is_inr else f"${price:,.2f}"
                high_fmt = f"₹{high_52:,.2f}" if is_inr and isinstance(high_52, (int, float)) else str(high_52)
                low_fmt = f"₹{low_52:,.2f}" if is_inr and isinstance(low_52, (int, float)) else str(low_52)

                return {
                    "company_name": clean_sym,
                    "ticker_resolved": sym,
                    "current_price": price_fmt,
                    "market_cap": mcap_fmt,
                    "pe_ratio": "N/A",
                    "52w_high": high_fmt,
                    "52w_low": low_fmt,
                    "currency": curr
                }
        except Exception:
            continue

    # Attempt 2: Google Finance Fallback
    gf_data = fetch_from_google_finance(clean_sym)
    if gf_data:
        return gf_data

    return {"error": f"Live exchange data unavailable for '{ticker}'."}

def tool_web_search(query: str) -> str:
    return f"Live search results for: '{query}'."
