import json
import yfinance as yf
from duckduckgo_search import DDGS

def fetch_usd_conversion_rate(currency: str) -> float:
    curr = currency.upper().strip()
    if curr == "USD":
        return 1.0
    try:
        fx = yf.Ticker(f"{curr}USD=X").fast_info.last_price
        if fx and fx > 0:
            return float(fx)
    except Exception:
        pass
    fallbacks = {
        "INR": 1.0 / 84.5,
        "EUR": 1.08,
        "GBP": 1.28,
        "JPY": 1.0 / 152.0,
        "CAD": 0.73,
        "AUD": 0.65
    }
    return fallbacks.get(curr, 1.0)

def tool_get_financial_data(ticker: str) -> str:
    try:
        sym = ticker.strip().upper()
        t = yf.Ticker(sym)
        fi = t.fast_info
        
        raw_currency = getattr(fi, 'currency', 'USD') or 'USD'
        raw_currency = raw_currency.upper().strip()

        price = getattr(fi, 'last_price', None) or getattr(fi, 'previous_close', 0.0)
        mcap = getattr(fi, 'market_cap', 0.0)
        fifty_high = getattr(fi, 'year_high', 'N/A')
        fifty_low = getattr(fi, 'year_low', 'N/A')

        sym_map = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
        native_sym = sym_map.get(raw_currency, f"{raw_currency} ")

        usd_rate = fetch_usd_conversion_rate(raw_currency)
        mcap_usd = (mcap * usd_rate) if isinstance(mcap, (int, float)) and mcap > 0 else None
        price_usd = (price * usd_rate) if isinstance(price, (int, float)) and price > 0 else None

        if raw_currency == "INR" and isinstance(mcap, (int, float)):
            cr_val = mcap / 1e7
            lakh_cr = mcap / 1e12
            local_mcap_str = f"₹{mcap:,.0f} (₹{cr_val:,.2f} Cr / ₹{lakh_cr:.2f} Lakh Cr)"
        else:
            local_mcap_str = f"{native_sym}{mcap:,.0f}" if isinstance(mcap, (int, float)) else "N/A"

        usd_mcap_str = (
            f"${mcap_usd / 1e9:,.2f} Billion USD" if (mcap_usd and mcap_usd >= 1e9)
            else (f"${mcap_usd / 1e6:,.2f} Million USD" if mcap_usd else "N/A")
        )

        metrics = {
            "Ticker": sym,
            "Native Currency": raw_currency,
            "Local Price": f"{native_sym}{price:,.2f}" if isinstance(price, (int, float)) else str(price),
            "Price (USD Equivalent)": f"${price_usd:,.2f}" if price_usd else "N/A",
            "Local Market Cap": local_mcap_str,
            "Normalized Market Cap (USD)": usd_mcap_str,
            "52W High (Local)": f"{native_sym}{fifty_high:.2f}" if isinstance(fifty_high, (int, float)) else str(fifty_high),
            "52W Low (Local)": f"{native_sym}{fifty_low:.2f}" if isinstance(fifty_low, (int, float)) else str(fifty_low)
        }
        # Compact JSON to keep token usage minimal
        return json.dumps(metrics)
    except Exception as e:
        return f"Market Data Error for {ticker}: {e}"

def tool_web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"No web results for '{query}'."
        return "\n".join([f"• [{r.get('title', '')}]: {r.get('body', '')[:250]}" for r in results])
    except Exception as e:
        return f"Search Error: {e}"

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": "Runs Python (numpy, pandas, matplotlib). Assign quantitative tables to DataFrame `df` to trigger automatic Excel export.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Executable Python code snippet."}},
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_data",
            "description": "Fetches current stock prices and market caps with automated currency conversion and USD normalization.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol (e.g. GS, MS, EVR, LAZ, SBIN.NS)."}},
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the live web for regulatory or corporate finance news.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query."}},
                "required": ["query"]
            }
        }
    }
]
