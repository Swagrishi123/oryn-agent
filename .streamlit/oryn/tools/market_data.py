"""Real-time Financial Market Data Engine (NSE, BSE, Global)."""
import re
from typing import Any, Dict, Optional
import requests
import yfinance as yf
from oryn.models.schemas import PermissionCategory, ToolResult
from oryn.tools.registry import BaseTool


class MarketDataTool(BaseTool):
    name = "get_market_data"
    description = "Fetch real-time stock price, 52-week range, and market cap."
    category = PermissionCategory.READ_ONLY
    timeout = 4.0

    def _fetch_yahoo_v8(self, symbol: str) -> Optional[Dict[str, Any]]:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36"}
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
        try:
            res = requests.get(url, headers=headers, timeout=2.5)
            if res.status_code == 200:
                meta = res.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
                price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
                if price:
                    high_52 = meta.get("fiftyTwoWeekHigh", "N/A")
                    low_52 = meta.get("fiftyTwoWeekLow", "N/A")
                    curr = meta.get("currency", "INR")
                    is_inr = curr == "INR" or ".NS" in symbol or ".BO" in symbol
                    return {
                        "price": f"₹{float(price):,.2f}" if is_inr else f"${float(price):,.2f}",
                        "high_52": f"₹{float(high_52):,.2f}" if is_inr and isinstance(high_52, (int, float)) else str(high_52),
                        "low_52": f"₹{float(low_52):,.2f}" if is_inr and isinstance(low_52, (int, float)) else str(low_52),
                        "currency": curr,
                        "symbol": symbol
                    }
        except Exception:
            pass
        return None

    def _fetch_dynamic_market_cap(self, symbol: str) -> str:
        try:
            t = yf.Ticker(symbol)
            mcap = getattr(t.fast_info, "market_cap", None)
            if mcap is None and hasattr(t.fast_info, "get"):
                mcap = t.fast_info.get("market_cap")
            
            if mcap and mcap > 0:
                if ".NS" in symbol or ".BO" in symbol:
                    return f"₹{mcap / 1e7:,.0f} Cr"
                elif mcap >= 1e12:
                    return f"${mcap / 1e12:.2f}T"
                else:
                    return f"${mcap / 1e9:.2f}B"
        except Exception:
            pass
        return "N/A"

    def execute(self, ticker: str) -> ToolResult:
        clean = ticker.strip().upper().replace("$", "")
        candidates = []
        if not clean.endswith((".NS", ".BO")):
            candidates.extend([f"{clean}.NS", f"{clean}.BO"])
        candidates.append(clean)

        for sym in candidates:
            chart = self._fetch_yahoo_v8(sym)
            if chart:
                mcap = self._fetch_dynamic_market_cap(sym)
                data = {
                    "symbol": chart["symbol"],
                    "current_price": chart["price"],
                    "52w_high": chart["high_52"],
                    "52w_low": chart["low_52"],
                    "market_cap": mcap,
                    "currency": chart["currency"]
                }
                return ToolResult(success=True, data=data, provenance="exchange_v8_fastinfo", is_live=True)

        return ToolResult(
            success=False,
            data=None,
            error=f"Instrument {clean} not found on exchange feeds.",
            is_live=False
        )
