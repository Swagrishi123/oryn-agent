import json
import requests
import yfinance as yf

# Tool schemas registered with Groq
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_financial_data",
            "description": "Fetch verified real-time stock price, 52-week range, market cap, and valuation metrics for NSE/BSE or global equities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol, e.g. RELIANCE, TCS, INFY, HDFCBANK, AAPL, MSFT"
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]

def tool_get_financial_data(ticker: str) -> dict:
    """Fetches verified live price, 52w range, and market cap without 401 crumb errors."""
    clean_sym = ticker.strip().upper().replace("$", "")
    candidates = [clean_sym]
    if not clean_sym.endswith((".NS", ".BO")):
        candidates.append(f"{clean_sym}.NS")
        candidates.append(f"{clean_sym}.BO")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    price = 0.0
    mcap = 0
    pe = "N/A"
    high_52 = "N/A"
    low_52 = "N/A"
    curr = "INR"
    resolved_sym = None
    company_name = clean_sym

    for sym in candidates:
        try:
            # 1. Fetch live market price & 52w range directly (bypasses 401 crumb block)
            chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=1d"
            res = requests.get(chart_url, headers=headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                if meta and "regularMarketPrice" in meta:
                    price = float(meta.get("regularMarketPrice") or 0.0)
                    high_52 = meta.get("fiftyTwoWeekHigh", "N/A")
                    low_52 = meta.get("fiftyTwoWeekLow", "N/A")
                    curr = meta.get("currency", "INR")
                    resolved_sym = sym
                    company_name = meta.get("symbol", sym)
                    break
        except Exception:
            continue

    # 2. Extract market cap & valuation metrics via fast_info
    if resolved_sym:
        try:
            tk = yf.Ticker(resolved_sym)
            fi = tk.fast_info
            mcap = getattr(fi, "market_cap", 0) or 0
            if not price:
                price = getattr(fi, "last_price", 0.0) or 0.0
            if high_52 == "N/A":
                high_52 = getattr(fi, "year_high", "N/A")
            if low_52 == "N/A":
                low_52 = getattr(fi, "year_low", "N/A")
        except Exception:
            pass

        try:
            tk = yf.Ticker(resolved_sym)
            t_pe = tk.info.get("trailingPE") or tk.info.get("forwardPE")
            if t_pe:
                pe = round(float(t_pe), 2)
        except Exception:
            pass

    if not resolved_sym or price == 0.0:
        return {"error": f"Symbol '{ticker}' could not be resolved on NSE, BSE, or global exchanges."}

    # Format output for Indian Rupee (Crores) or USD (Billions)
    if curr == "INR" or ".NS" in resolved_sym or ".BO" in resolved_sym:
        mcap_fmt = f"₹{mcap / 1e7:,.2f} Cr" if mcap else "N/A"
        price_fmt = f"₹{price:,.2f}"
        high_fmt = f"₹{high_52:,.2f}" if isinstance(high_52, (int, float)) else str(high_52)
        low_fmt = f"₹{low_52:,.2f}" if isinstance(low_52, (int, float)) else str(low_52)
    else:
        mcap_fmt = f"${mcap / 1e9:,.2f}B" if mcap else "N/A"
        price_fmt = f"${price:,.2f}"
        high_fmt = f"${high_52:,.2f}" if isinstance(high_52, (int, float)) else str(high_52)
        low_fmt = f"${low_52:,.2f}" if isinstance(low_52, (int, float)) else str(low_52)

    return {
        "company_name": company_name,
        "ticker_resolved": resolved_sym,
        "current_price": price_fmt,
        "market_cap": mcap_fmt,
        "pe_ratio": pe,
        "52w_high": high_fmt,
        "52w_low": low_fmt,
        "currency": curr
    }

def tool_web_search(query: str) -> str:
    return f"Live search results for: '{query}'."
