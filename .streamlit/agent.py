"""ORYN Analyst Autonomous Orchestrator with Centralized Guardrails."""
import re
import time
from typing import Optional
from oryn.llm import GroqProvider
from oryn.models.schemas import (
    ActionProposal,
    AgentResponse,
    ApprovalDecision,
    ApprovalStatus,
    PermissionCategory
)
from oryn.policy import PolicyEngine
from oryn.state import SessionState
from oryn.tools.dcf import DCFModelTool
from oryn.tools.market_data import MarketDataTool
from oryn.tools.registry import ToolRegistry
from oryn.tools.web_search import WebSearchTool


class OrynAgent:
    SYSTEM_PROMPT = (
        "You are ORYN Analyst, an elite institutional BFSI quantitative analyst and corporate intelligence advisor.\n\n"
        "CRITICAL CONTEXT & FACT PROTOCOL:\n"
        "- When [VERIFIED LIVE EXCHANGE DATA] or [VERIFIED RESEARCH RECORDS] is provided, "
        "base your response strictly on those verified facts. Never contradict the context.\n"
        "- If asked who founded a company or holds public office, state the exact names and dates directly from the verified records.\n\n"
        "CRITICAL FORMATTING & TABLE RULES (IMMUNITY TO KATEX COLLAPSE):\n"
        "1. DIRECT STRUCTURAL DELIVERY: Eliminate introductory pleasantries. Begin directly with the analysis header.\n"
        "2. NO DOLLAR SIGNS IN TABLES: NEVER use '$' inside table cells or column headers. "
        "State currency in headers as 'USD M' or 'INR'. Cell values MUST be pure numbers (e.g., | 1 | 60.00 | 15.00 |).\n"
        "3. STRICT MARKDOWN TABLE SYNTAX: Every table row MUST start and end with a pipe (|). "
        "Always place the divider row (| :--- | :--- |) immediately below the header row. Never output raw HTML table tags.\n\n"
        "FINANCIAL REASONING & WATERFALL PROTOCOLS:\n"
        "1. CHAPTER 11 RESTRUCTURING WATERFALL:\n"
        "   - Calculate claims strictly in linear sequence: DIP -> First Lien -> Second Lien -> Senior Unsecured -> Subordinated.\n"
        "   - Present the distribution in ONE unified Markdown table. Never debate, recalculate, or re-evaluate the table in subsequent paragraphs.\n"
        "   - The fulcrum security is the tranche that receives a partial recovery.\n"
        "2. FIG & ALM BANKING DYNAMICS:\n"
        "   - A policy rate cut on an asset-sensitive bank causes Net Interest Margin (NIM) COMPRESSION because floating assets reprice down faster than liabilities.\n"
        "   - High deposit beta cushions NIM compression (deposit costs drop quickly). Low deposit beta causes severe NIM compression.\n"
        "   - Falling benchmark yields increase bond prices, reversing unrealized losses in Available-For-Sale (AFS) portfolios, which directly EXPANDS Tangible Common Equity (TCE).\n"
        "3. DCF MODELING: Execute quantitative schedules, render matplotlib artifacts, and write 'model.xlsx'."
    )

    KNOWN_ENTITIES = {
        "reliance": "RELIANCE",
        "tcs": "TCS",
        "tata consultancy": "TCS",
        "infosys": "INFY",
        "infy": "INFY",
        "hdfc": "HDFCBANK",
        "hdfc bank": "HDFCBANK",
        "icici": "ICICIBANK",
        "icici bank": "ICICIBANK",
        "sbi": "SBIN",
        "state bank of india": "SBIN",
        "tata motors": "TATAMOTORS",
        "itc": "ITC",
        "wipro": "WIPRO",
        "bharti airtel": "BHARTIARTL",
        "airtel": "BHARTIARTL",
        "apple": "AAPL",
        "microsoft": "MSFT",
        "nvidia": "NVDA",
        "google": "GOOGL",
        "tesla": "TSLA"
    }

    FINANCIAL_ACRONYM_BLACKLIST = {
        "NIM", "ALM", "TCE", "AFS", "FIG", "BPS", "DIP", "LBO", "WACC", "DCF",
        "EBITDA", "FCF", "APR", "EV", "PE", "IRR", "CAGR", "ROE", "ROA", "ROIC",
        "EPS", "NAV", "PBR", "PER", "DPS", "RBI", "SEBI", "FED", "CEO", "CFO",
        "USD", "INR", "NSE", "BSE"
    }

    def __init__(self, api_key: str, model_name: Optional[str] = None, hitl_enabled: bool = True):
        self.llm = GroqProvider(api_key=api_key, preferred_model=model_name)
        self.policy = PolicyEngine(hitl_enabled=hitl_enabled)
        self.state = SessionState()
        self.registry = ToolRegistry()

        self.registry.register(MarketDataTool())
        self.registry.register(DCFModelTool())
        self.registry.register(WebSearchTool())

    def _sanitize_output(self, text: str) -> str:
        cleaned = re.sub(r'```(?:python)?\s*[\s\S]*?```', '', text)
        lines = cleaned.split('\n')
        output = []
        current_table = []

        def flush_tbl(tbl):
            if not tbl:
                return []
            if len(tbl) >= 2 and not any('---' in r for r in tbl[:2]):
                cols = [c.strip() for c in tbl[0].strip('|').split('|')]
                tbl.insert(1, "| " + " | ".join([":---"] * len(cols)) + " |")
            return [""] + tbl + [""]

        for line in lines:
            stripped = line.strip()
            stripped = re.sub(r'^\d+[\.\)]\s*\|', '|', stripped)
            if '|' in stripped:
                if not stripped.startswith('|'):
                    stripped = '| ' + stripped
                if not stripped.endswith('|'):
                    stripped = stripped + ' |'
                stripped = stripped.replace('$', 'USD ')
                current_table.append(stripped)
            else:
                if current_table:
                    output.extend(flush_tbl(current_table))
                    current_table = []
                output.append(line)

        if current_table:
            output.extend(flush_tbl(current_table))

        res = '\n'.join(output)
        res = re.sub(r'\$\s*([0-9])', r'USD \1', res)
        return re.sub(r'\n{3,}', '\n\n', res).strip()

    def process(self, user_input: str, approval_decision: Optional[ApprovalDecision] = None) -> AgentResponse:
        start_time = time.perf_counter()
        p_lower = user_input.lower()

        # 1. Deterministic DCF branch
        if any(k in p_lower for k in ["dcf", "discounted cash flow", "unlevered free cash flow"]):
            dcf_tool: DCFModelTool = self.registry.get("calculate_dcf")
            result = dcf_tool.execute()
            if result.success:
                return AgentResponse(
                    content=self._sanitize_output(result.data["briefing"]),
                    actions_executed=["calculate_dcf"],
                    plot_bytes=result.data["plot_bytes"],
                    excel_bytes=result.data["excel_bytes"],
                    execution_time=time.perf_counter() - start_time
                )
            return AgentResponse(content=f"Error executing DCF model: {result.error}")

        # 2. Intent-Gated Equity Ticker Discovery (handles Title Case and Known Entities)
        found_tickers = []
        for name, sym in self.KNOWN_ENTITIES.items():
            if re.search(r'\b' + re.escape(name) + r'\b', p_lower):
                if sym not in found_tickers:
                    found_tickers.append(sym)

        is_stock_query = any(k in p_lower for k in ["price", "market cap", "52-week", "share", "stock", "quote", "numbers"])
        if is_stock_query:
            for match in re.findall(r'\b[A-Z]{2,10}\b', user_input):
                if match not in self.FINANCIAL_ACRONYM_BLACKLIST and match not in found_tickers:
                    found_tickers.append(match)

        injected_context = []
        actions_taken = []

        if found_tickers:
            market_tool: MarketDataTool = self.registry.get("get_market_data")
            rows = []
            for t in found_tickers[:3]:
                res = market_tool.execute(t)
                if res.success:
                    d = res.data
                    rows.append(f"- {d['symbol']}: Price = {d['current_price']}, Cap = {d['market_cap']}, 52W High = {d['52w_high']}, 52W Low = {d['52w_low']}")
                    actions_taken.append(f"get_market_data({t})")
            if rows:
                injected_context.append("[VERIFIED LIVE EXCHANGE DATA - USE IN TABLE]:\n" + "\n".join(rows))

        # 3. Autonomous Web Research
        if not found_tickers and not any(k in p_lower for k in ["waterfall", "restructuring", "ebitda margin", "wacc"]):
            search_tool: WebSearchTool = self.registry.get("web_search")
            s_res = search_tool.execute(user_input)
            if s_res.success:
                injected_context.append(f"[VERIFIED RESEARCH RECORDS]:\n{s_res.data}")
                actions_taken.append("web_search")

        # 4. LLM Generation
        augmented_prompt = user_input
        if injected_context:
            augmented_prompt += "\n\n" + "\n\n".join(injected_context)

        self.state.add_message("user", augmented_prompt)
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + self.state.get_context_window()

        try:
            raw_reply = self.llm.complete(messages, max_tokens=800)
            cleaned_reply = self._sanitize_output(raw_reply)
            self.state.add_message("assistant", cleaned_reply)

            return AgentResponse(
                content=cleaned_reply,
                actions_executed=actions_taken,
                execution_time=time.perf_counter() - start_time
            )
        except Exception as e:
            return AgentResponse(
                content=f"Error executing agent reasoning: {str(e)}",
                execution_time=time.perf_counter() - start_time
            )
