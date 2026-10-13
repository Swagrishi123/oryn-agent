import json
import yfinance as yf

# Tool definition schemas for Groq / OpenAI tool calling
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": "Executes quantitative calculations, financial modeling schedules, DCF formulas, and exports Excel models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Valid Python code to execute."
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_data",
            "description": "Fetches real-time market data, price, P/E ratio, market cap, and EBITDA for global and Indian stocks (NSE/BSE).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g., 'AAPL', 'RELIANCE', 'TCS', 'HDFCBANK'."
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]

def tool_get_financial_data(ticker: str) -> dict:
    """
    Fetches real-time financial metrics with automatic suffix fallback
    for Indian stock exchanges (.NS for NSE, .BO for BSE) and US tickers.
    """
    clean_sym = ticker.strip().upper()
    
    # Priority search order: exact ticker -> NSE suffix -> BSE suffix
    candidates = [clean_sym]
    if not clean_sym.endswith(('.NS', '.BO')):
        candidates.append(f"{clean_sym}.NS")
        candidates.append(f"{clean_sym}.BO")

    info = {}
    resolved_ticker = None

    for sym in candidates:
        try:
            t = yf.Ticker(sym)
            data = t.info
            # Validate that data actually came back
            if data and ('regularMarketPrice' in data or 'currentPrice' in data):
                info = data
                resolved_ticker = sym
                break
        except Exception:
            continue

    if not info or not resolved_ticker:
        return {"error": f"Symbol '{ticker}' could not be resolved on US, NSE, or BSE exchanges."}

    currency = info.get("currency", "INR" if (".NS" in resolved_ticker or ".BO" in resolved_ticker) else "USD")
    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
    mcap = info.get("marketCap", 0)
    ebitda = info.get("ebitda", 0)
    pe = info.get("trailingPE") or info.get("forwardPE") or 0.0

    return {
        "ticker_resolved": resolved_ticker,
        "currency": currency,
        "current_price": round(price, 2),
        "market_cap": f"₹{mcap / 1e7:,.2f} Cr" if currency == "INR" else f"${mcap / 1e9:,.2f}B",
        "pe_ratio": round(pe, 2) if pe else "N/A",
        "ebitda": f"₹{ebitda / 1e7:,.2f} Cr" if currency == "INR" else f"${ebitda / 1e9:,.2f}B",
        "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
        "company_name": info.get("longName", clean_sym)
    }

def tool_web_search(query: str) -> str:
    """Fallback search function for general company background queries."""
    return f"Live search results for: '{query}'."
