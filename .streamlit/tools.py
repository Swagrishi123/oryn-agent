import json
import yfinance as yf

# Schemas telling Groq what tools exist
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_financial_data",
            "description": "Fetch real-time stock price, P/E ratio, market cap, and EBITDA for any Indian (NSE/BSE) or global stock ticker.",
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
    """Fetches real-time price, market cap, and valuation metrics for Indian and global equities."""
    clean_ticker = ticker.strip().upper().replace("$", "")
    
    # Priority order: As provided -> NSE (.NS) -> BSE (.BO)
    candidates = [clean_ticker]
    if not clean_ticker.endswith((".NS", ".BO")):
        candidates.append(f"{clean_ticker}.NS")
        candidates.append(f"{clean_ticker}.BO")

    data = None
    resolved_sym = None

    for sym in candidates:
        try:
            tk = yf.Ticker(sym)
            info = tk.info
            # Validate that real data came back
            if info and ("currentPrice" in info or "regularMarketPrice" in info):
                data = info
                resolved_sym = sym
                break
        except Exception:
            continue

    if not data or not resolved_sym:
        return {"error": f"Could not find live market data for ticker '{ticker}' on NSE, BSE, or global exchanges."}

    curr = data.get("currency", "INR" if (".NS" in resolved_sym or ".BO" in resolved_sym) else "USD")
    price = data.get("currentPrice") or data.get("regularMarketPrice") or 0.0
    mcap = data.get("marketCap", 0)
    ebitda = data.get("ebitda", 0)
    pe = data.get("trailingPE") or data.get("forwardPE") or "N/A"

    # Format market cap and EBITDA in Crores for Indian stocks, or Billions for US stocks
    if curr == "INR":
        mcap_formatted = f"₹{mcap / 1e7:,.2f} Cr" if mcap else "N/A"
        ebitda_formatted = f"₹{ebitda / 1e7:,.2f} Cr" if ebitda else "N/A"
        price_formatted = f"₹{price:,.2f}"
    else:
        mcap_formatted = f"${mcap / 1e9:,.2f}B" if mcap else "N/A"
        ebitda_formatted = f"${ebitda / 1e9:,.2f}B" if ebitda else "N/A"
        price_formatted = f"${price:,.2f}"

    return {
        "company_name": data.get("longName", resolved_sym),
        "ticker_resolved": resolved_sym,
        "current_price": price_formatted,
        "market_cap": mcap_formatted,
        "pe_ratio": round(pe, 2) if isinstance(pe, (int, float)) else pe,
        "ebitda": ebitda_formatted,
        "currency": curr,
        "52w_high": data.get("fiftyTwoWeekHigh", "N/A"),
        "52w_low": data.get("fiftyTwoWeekLow", "N/A")
    }

def tool_web_search(query: str) -> str:
    return f"Live search results for: '{query}'."
