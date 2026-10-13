import io
import json
import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
from tools import AGENT_TOOLS, tool_get_financial_data, tool_web_search
from sandbox import tool_python_interpreter

SYSTEM_PROMPT = (
    "You are ORYN Analyst, an elite institutional BFSI quantitative analyst and investment banking advisor.\n\n"
    "OPERATING & FORMATTING RULES:\n"
    "1. DIRECT STRUCTURAL DELIVERY: Eliminate all pleasantries, greetings, and meta-commentary. Begin immediately with the analysis.\n"
    "2. ADAPTIVE PRESENTATION:\n"
    "   - For Deal Modeling, M&A, FIG, Restructuring, and Financial Metrics: Use structured Markdown tables and formal section headers.\n"
    "   - For Biographies, Macro Overviews, Political Context, and General Concepts: Use clean executive prose and bullet points. DO NOT force artificial tables or parameter matrices for simple biographies.\n"
    "3. STRICT MARKDOWN TABLE SYNTAX (WHEN TABLES ARE USED):\n"
    "   - Every table row MUST be separated by an explicit newline (\\n).\n"
    "   - Always place the divider row (| :--- | :--- |) on its own separate line between the header and data.\n"
    "   - Never concatenate table cells or rows into a single run-on sentence.\n"
    "4. FIG & BANKING REASONING:\n"
    "   - In rate cuts, asset-sensitive bank NIM compresses. High deposit beta cushions margin; low deposit beta causes severe compression.\n"
    "   - Rate cuts increase bond prices (reversing AFS paper losses to expand TCE).\n"
    "5. LEVERAGED FINANCE & M&A:\n"
    "   - Junior/unsecured debt must price higher than senior secured debt.\n"
    "   - Sources & Uses percentages must sum to 100.0%.\n"
    "   - For stock deals, EPS accretion depends on Buyer P/E vs Target P/E; debt deals depend on Target Earnings Yield vs After-Tax Debt Cost.\n"
    "6. ARITHMETIC PROTOCOL: Always calculate step-by-step to prevent hallucinations.\n"
    "7. DCF VALUATIONS: Execute formulas, generate visual charts, and export 'model.xlsx' only for explicit quantitative modeling requests."
)
CACHED_WORKING_MODEL = None

def is_quantitative_model_request(prompt: str) -> bool:
    """Distinguishes explicit DCF model generation from standard advisory memos."""
    p = prompt.lower()
    dcf_terms = [
        "dcf", "discounted cash flow", "sensitivity matrix", 
        "unlevered", "run a financial model", 
        "build a financial model", "run a 5-year", "run a 10-year",
        "export the financial model to excel"
    ]
    if any(t in p for t in dcf_terms):
        return True
        
    has_discount_metric = any(m in p for m in ["wacc", "discount rate", "terminal growth"])
    has_financial_metric = any(m in p for m in ["ebitda margin", "revenue growth", "cash flow projection"])
    return bool(has_discount_metric and has_financial_metric)

def extract_code_from_markdown(text: str) -> str | None:
    m = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m_generic = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if m_generic:
        return m_generic.group(1).strip()
    return None

def sanitize_report_formatting(text: str) -> str:
    """Restores linebreaks around Markdown tables so Streamlit never renders them as run-on text."""
    cleaned = re.sub(r'```(?:python)?\s*[\s\S]*?```', '', text)
    cleaned = re.sub(r'([^\n])\n(\|.*\|)', r'\1\n\n\2', cleaned)
    cleaned = re.sub(r'(\|.*\|)\n([^\n|])', r'\1\n\n\2', cleaned)
    cleaned = re.sub(r'(?<!\\)\$', r'\$', cleaned)
    return re.sub(r'\n{3,}', '\n\n', cleaned).strip()

def enforce_sandbox_contracts(code_str: str) -> str:
    clean_code = re.sub(r'ExcelWriter\s*\(\s*[\'"][^\'"]+\.xlsx[\'"]', 'ExcelWriter("model.xlsx"', code_str)
    clean_code = re.sub(r'\.to_excel\s*\(\s*[\'"][^\'"]+\.xlsx[\'"]', '.to_excel("model.xlsx"', clean_code)
    preamble = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "years = 5\n"
    )
    return preamble + "\n" + clean_code

def resolve_api_key(api_key: str = "") -> str:
    if api_key and api_key.strip():
        return api_key.strip()
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()

def get_best_available_model(client: Groq, exclude_model: str = None, force_refresh: bool = False) -> str:
    global CACHED_WORKING_MODEL
    if CACHED_WORKING_MODEL and not force_refresh and CACHED_WORKING_MODEL != exclude_model:
        return CACHED_WORKING_MODEL

    try:
        models_data = client.models.list().data
        available_ids = [m.id for m in models_data]
    except Exception:
        available_ids = []

    filter_out = ["safeguard", "guard", "whisper", "orpheus", "allam", "arabic", "vision", "embed", "audio"]
    candidates = [mid for mid in available_ids if not any(b in mid.lower() for b in filter_out) and mid != exclude_model]
    preference = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.2-3b-preview", "llama-3.1-8b-instant"]

    test_queue = [m for m in preference if m in candidates] + [m for m in candidates if m not in preference]
    if not test_queue:
        test_queue = candidates if candidates else preference

    for model_name in test_queue:
        try:
            client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": "ping"}], max_tokens=1)
            CACHED_WORKING_MODEL = model_name
            return model_name
        except Exception:
            continue

    if candidates:
        CACHED_WORKING_MODEL = candidates[0]
        return CACHED_WORKING_MODEL

    raise RuntimeError("Could not connect to any active Groq text model.")

def create_completion_with_fallback(client: Groq, preferred_model: str = None, status_callback=None, **kwargs):
    global CACHED_WORKING_MODEL
    active_model = preferred_model or get_best_available_model(client)
    kwargs["max_tokens"] = min(kwargs.get("max_tokens", 850), 850)
    kwargs["temperature"] = kwargs.get("temperature", 0.1)
    kwargs["model"] = active_model

    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "rate_limit_exceeded" in err_str or "output tokens per minute" in err_str:
            delay_match = re.search(r'try again in ([0-9.]+)s', err_str)
            sleep_sec = float(delay_match.group(1)) + 0.5 if delay_match else 2.5
            if status_callback:
                status_callback("info", f"Quota reset pause ({sleep_sec:.1f}s)... retrying on secondary model.")
            time.sleep(sleep_sec)
            alt_model = get_best_available_model(client, exclude_model=active_model, force_refresh=True)
            kwargs["model"] = alt_model
            kwargs["max_tokens"] = 600
            try:
                return client.chat.completions.create(**kwargs)
            except Exception:
                kwargs["model"] = active_model
                kwargs["max_tokens"] = 500
                return client.chat.completions.create(**kwargs)

        if any(term in err_str for term in ["404", "model_not_found", "decommissioned", "400"]):
            alt_model = get_best_available_model(client, exclude_model=active_model, force_refresh=True)
            kwargs["model"] = alt_model
            return client.chat.completions.create(**kwargs)
        raise e

def build_contextual_messages(user_input: str, system_prompt: str, max_turns: int = 3) -> list[dict]:
    history = []
    try:
        import streamlit as st
        if "messages" in st.session_state and isinstance(st.session_state.messages, list):
            candidates = st.session_state.messages[:-1] if st.session_state.messages[-1].get("content") == user_input else st.session_state.messages
            for m in candidates[-(max_turns * 2):]:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role in ["user", "assistant"] and content:
                    history.append({"role": role, "content": str(content)[:250] + ("..." if len(content) > 250 else "")})
    except Exception:
        pass
    return [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_input}]

def parse_prompt_financial_parameters(prompt: str) -> tuple[float, float, float, float, float, int]:
    p = prompt.lower()
    rev_match = re.search(r'\$?(\d+(?:\.\d+)?)\s*(?:m|million)?\s*(?:base\s*)?rev', p)
    base_rev = float(rev_match.group(1)) if rev_match else 50.0

    growth_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:annual\s*)?growth', p)
    growth = float(growth_match.group(1)) / 100.0 if growth_match else 0.20

    margin_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*ebitda', p)
    margin = float(margin_match.group(1)) / 100.0 if margin_match else 0.25

    wacc_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*wacc', p)
    wacc = float(wacc_match.group(1)) / 100.0 if wacc_match else 0.10

    tg_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*terminal\s*(?:growth)?', p)
    tg = float(tg_match.group(1)) / 100.0 if tg_match else 0.025

    return base_rev, growth, margin, wacc, tg, 5

def df_to_markdown_table(df: pd.DataFrame, include_index: bool = False) -> str:
    headers = (["Parameter"] if include_index else []) + [str(col) for col in df.columns]
    rows = [[str(idx)] + [str(val) for val in row] for idx, row in zip(df.index, df.values)] if include_index else [[str(val) for val in row] for row in df.values]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n\n" + "\n".join([header_line, sep_line] + row_lines) + "\n\n"

def generate_deterministic_dcf_artifacts(user_input: str) -> tuple[str, bytes, bytes]:
    base_rev, growth, margin, wacc, tg, years = parse_prompt_financial_parameters(user_input)
    years_idx = list(range(1, years + 1))

    rev = [base_rev * ((1 + growth) ** t) for t in years_idx]
    ebitda = [r * margin for r in rev]
    ufcf = ebitda
    df = [1 / ((1 + wacc) ** t) for t in years_idx]
    pv_ufcf = [u * d for u, d in zip(ufcf, df)]
    sum_pv_ufcf = sum(pv_ufcf)

    tv_ufcf = ufcf[-1] * (1 + tg)
    tv = tv_ufcf / (wacc - tg)
    pv_tv = tv / ((1 + wacc) ** years)
    ev = sum_pv_ufcf + pv_tv

    wacc_range = [wacc - 0.02, wacc - 0.01, wacc, wacc + 0.01, wacc + 0.02]
    tg_range = [tg - 0.01, tg - 0.005, tg, tg + 0.005, tg + 0.01]
    sens_data = {}
    for w in wacc_range:
        col = []
        pv_cf = sum([u / ((1 + w) ** t) for t, u in enumerate(ufcf, 1)])
        for g in tg_range:
            cur_tv = (ufcf[-1] * (1 + g)) / (w - g)
            cur_pv_tv = cur_tv / ((1 + w) ** years)
            col.append(f"${pv_cf + cur_pv_tv:.2f}M")
        sens_data[f"WACC {w*100:.1f}%"] = col

    df_sens = pd.DataFrame(sens_data, index=[f"TG {g*100:.1f}%" for g in tg_range])
    df_proj = pd.DataFrame({
        "Year": [str(y) for y in years_idx],
        "Revenue ($M)": [f"${x:.2f}M" for x in rev],
        "EBITDA ($M)": [f"${x:.2f}M" for x in ebitda],
        "UFCF ($M)": [f"${x:.2f}M" for x in ufcf],
        "Discount Factor": [f"{x:.4f}" for x in df],
        "PV of UFCF ($M)": [f"${x:.2f}M" for x in pv_ufcf]
    })

    fig, ax = plt.subplots(figsize=(8, 4.5))
    components = ['PV of Cash Flows', 'PV of Terminal Value']
    values = [sum_pv_ufcf, pv_tv]
    bars = ax.bar(components, values, color=['#1f77b4', '#2ca02c'], width=0.45)
    ax.set_title(f'Enterprise Value Breakdown ($M) — Total: ${ev:.2f}M', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Present Value ($M)')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'${yval:.2f}M', ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=200)
    plt.close(fig)
    plot_bytes = img_buf.getvalue()

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
        df_proj.to_excel(writer, sheet_name='DCF_Projections', index=False)
        pd.DataFrame({
            "Metric": ["PV of 5-Yr Cash Flows ($M)", "Terminal Value ($M)", "PV of Terminal Value ($M)", "Enterprise Value ($M)"],
            "Value": [round(sum_pv_ufcf, 2), round(tv, 2), round(pv_tv, 2), round(ev, 2)]
        }).to_excel(writer, sheet_name='Valuation_Summary', index=False)
        df_sens.to_excel(writer, sheet_name='Sensitivity_Matrix')
    excel_bytes = excel_buf.getvalue()

    summary_df = pd.DataFrame({
        "Metric": ["Base Revenue", "Annual Growth", "EBITDA Margin", "WACC", "Terminal Growth Rate", "Enterprise Value"],
        "Value": [f"${base_rev:.2f}M", f"{growth*100:.1f}%", f"{margin*100:.1f}%", f"{wacc*100:.1f}%", f"{tg*100:.2f}%", f"${ev:.2f}M"]
    })

    calc_briefing = (
        "=====================================================================\n"
        "INSTITUTIONAL VALUATION MEMORANDUM | DCF SCHEDULE\n"
        "=====================================================================\n\n"
        f"**Core Valuation Summary:**\n{df_to_markdown_table(summary_df, include_index=False)}\n\n"
        f"**1. 5-Year Discrete Cash Flow Projections:**\n{df_to_markdown_table(df_proj, include_index=False)}\n\n"
        f"**2. Sensitivity Matrix (Enterprise Value in $M):**\n{df_to_markdown_table(df_sens, include_index=True)}"
    )

    return calc_briefing, plot_bytes, excel_bytes

def run_analyst_agent(
    user_input: str,
    api_key: str = "",
    selected_model: str = None,
    engine_mode: str = "Cloud (Groq BYOK)",
    hitl_enabled: bool = True,
    status_callback=None,
    **kwargs
) -> tuple[str, bytes | None, bytes | None, float]:
    start_time = time.perf_counter()
    active_key = resolve_api_key(api_key)
    if not active_key:
        return "⚠️ **API Key Missing**: Please provide a Groq API key in the sidebar.", None, None, 0.0

    try:
        client = Groq(api_key=active_key, max_retries=0)
    except Exception as e:
        return f"⚠️ **Client Error**: {e}", None, None, 0.0

    # Auto-resolve live market data if the user asks for a ticker quote
    ticker_match = re.search(r'\b(?:quote|stock|price of|pe of|market cap of)\s+([A-Za-z0-9\.\-\&]+)', user_input.lower())
    extra_financial_context = ""
    if ticker_match:
        extracted_sym = ticker_match.group(1).upper()
        if status_callback:
            status_callback("info", f"Fetching live market metrics for {extracted_sym}...")
        live_data = tool_get_financial_data(extracted_sym)
        if "error" not in live_data:
            extra_financial_context = f"\n[VERIFIED LIVE MARKET DATA: {json.dumps(live_data)}]\n"

    enriched_input = user_input + extra_financial_context
    is_dcf_calc = is_quantitative_model_request(enriched_input)

    if not is_dcf_calc:
        try:
            messages = build_contextual_messages(enriched_input, SYSTEM_PROMPT, max_turns=3)
            resp = create_completion_with_fallback(
                client=client,
                preferred_model=selected_model,
                status_callback=status_callback,
                messages=messages,
                max_tokens=850,
                temperature=0.1
            )
            elapsed = time.perf_counter() - start_time
            content = resp.choices[0].message.content or ""
            return sanitize_report_formatting(content), None, None, elapsed
        except Exception as e:
            return f"Error: {e}", None, None, time.perf_counter() - start_time

    # DCF Modeling Engine
    generated_plot = None
    generated_excel = None
    structured_briefing = ""

    if status_callback:
        status_callback("info", "Executing quantitative DCF model...")
    briefing_text, plot_bytes, excel_bytes = generate_deterministic_dcf_artifacts(enriched_input)
    generated_plot = plot_bytes
    generated_excel = excel_bytes
    structured_briefing = briefing_text

    final_text = sanitize_report_formatting(structured_briefing)
    elapsed = time.perf_counter() - start_time
    return final_text, generated_plot, generated_excel, elapsed
