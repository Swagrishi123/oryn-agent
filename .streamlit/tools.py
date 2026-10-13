import json
import re
import requests
import yfinance as yf

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_financial_data",
            "description": "Fetch real-time stock price, 52-week range, and market cap for Indian (BSE/NSE) and global equities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol or company name, e.g. RELIANCE, TCS, INFY, HDFCBANK, AAPL"
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]

# Official BSE scrip codes for prominent Indian equities
BSE_SCRIP_REGISTRY = {
    "RELIANCE": "500325",
    "TCS": "532540",
    "INFY": "500209",
    "INFOSYS": "500209",
    "HDFCBANK": "500180",
    "HDFC": "500180",
    "ICICIBANK": "532174",
    "SBIN": "500112",
    "SBI": "500112",
    "TATAMOTORS": "500570",
    "ITC": "500875",
    "BHARTIARTL": "532454",
    "AIRTEL": "532454",
    "WIPRO": "507685",
    "HINDUNILVR": "500696",
    "LT": "500510",
    "KOTAKBANK": "500247",
    "AXISBANK": "532215",
    "BAJFINANCE": "500034",
    "MARUTI": "532500"
}

def fetch_bse_live_quote(scrip_code: str, display_name: str) -> dict | None:
    """Fetches real-time price, market cap, and 52-week range directly from BSE India API."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Referer": "https://www.bseindia.com/",
        "Accept": "application/json, text/plain, */*"
    }
    url = f"https://api.bseindia.com/BseIndiaAPI/api/getScripHeaderData/w?Debtflag=&scripcode={scrip_code}&seriesid="
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            header = data.get("Header", {})
            curr_rate = header.get("CurrRate")
            if curr_rate:
                price_val = float(str(curr_rate).replace(",", "").strip())
                prev_close = header.get("PrevClose", "N/A")
                high_52 = header.get("52High", "N/A")
                low_52 = header.get("52Low", "N/A")
                mcap = header.get("MarketCapFull") or header.get("MarketCapFreeFloat") or "N/A"

                return {
                    "company_name": display_name,
                    "ticker_resolved": f"BSE:{scrip_code}",
                    "current_price": f"₹{price_val:,.2f}",
                    "market_cap": f"₹{mcap} Cr" if mcap != "N/A" and "Cr" not in str(mcap) else str(mcap),
                    "52w_high": f"₹{high_52}" if high_52 != "N/A" else "N/A",
                    "52w_low": f"₹{low_52}" if low_52 != "N/A" else "N/A",
                    "previous_close": f"₹{prev_close}",
                    "currency": "INR"
                }
    except Exception:
        pass
    return None

def tool_get_financial_data(ticker: str) -> dict:
    """Universal financial data resolver supporting Indian stocks via BSE and global tickers via Yahoo."""
    clean_sym = ticker.strip().upper().replace("$", "")

    # Priority 1: Indian Equities via official BSE API
    if clean_sym in BSE_SCRIP_REGISTRY:
        bse_data = fetch_bse_live_quote(BSE_SCRIP_REGISTRY[clean_sym], clean_sym)
        if bse_data:
            return bse_data

    # Priority 2: Global Tickers via Yahoo Fast Info
    try:
        tk = yf.Ticker(clean_sym)
        fi = tk.fast_info
        price = getattr(fi, "last_price", 0.0) or 0.0
        if price > 0.0:
            mcap = getattr(fi, "market_cap", 0) or 0
            curr = getattr(fi, "currency", "USD")
            return {
                "company_name": clean_sym,
                "ticker_resolved": clean_sym,
                "current_price": f"${price:,.2f}",
                "market_cap": f"${mcap / 1e9:,.2f}B" if mcap else "N/A",
                "52w_high": f"${getattr(fi, 'year_high', 0.0):,.2f}",
                "52w_low": f"${getattr(fi, 'year_low', 0.0):,.2f}",
                "currency": curr
            }
    except Exception:
        pass

    return {"error": f"Live data feed currently unavailable for '{ticker}'."}

def tool_web_search(query: str) -> str:
    return f"Live search results for: '{query}'."
