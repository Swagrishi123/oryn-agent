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
    "You are ORYN Analyst, an elite institutional BFSI quantitative analyst and investment banking advisor.\n"
    "CRITICAL OPERATING & FORMATTING RULES:\n"
    "1. DIRECT STRUCTURAL DELIVERY: Eliminate all introductory greetings, pleasantries, and meta-commentary. "
    "Begin immediately with the report title.\n"
    "2. STRICT MARKDOWN TABLE SYNTAX: Every table row MUST be separated by an actual newline (\\n) and start/end with a pipe (|). "
    "Never concatenate table rows into run-on paragraphs. Follow this exact Markdown table structure:\n"
    "| Col 1 | Col 2 |\n"
    "| :--- | :--- |\n"
    "| Val 1 | Val 2 |\n"
    "3. INSTITUTIONAL FIG & BANKING REASONING:\n"
    "   - Asset-Sensitive Banks: In a RATE-CUT environment, loan yields reprice downward faster than deposit costs can be cut. "
    "Therefore, Net Interest Margin (NIM) strictly COMPRESSES (Asset Yield Delta - Liability Cost Delta is negative).\n"
    "   - Deposit Beta Direction in Rate Cuts:\n"
    "     * Low Deposit Beta (e.g., 30%): Deposit costs are sticky downward (-30 bps relief), causing SEVERE NIM COMPRESSION (-45 bps).\n"
    "     * High Deposit Beta (e.g., 60%): Deposit costs drop aggressively (-60 bps relief), CUSHIONING margins and limiting NIM compression (-15 bps).\n"
    "   - NII Impact: Asset Yield Drag minus Liability Cost Relief multiplied by Earning Assets. Never report positive NIM expansion for a rate cut on an asset-sensitive bank.\n"
    "   - Secondary Rate Cut Effects: Declining yields cause bond prices to RISE (reversing AFS paper losses and boosting TCE). NIB deposit opportunity cost FALLS (slowing deposit flight to money markets).\n"
    "4. M&A ACCRETION REASONING:\n"
    "   - Stock deals depend on relative P/E arbitrage (Buyer P/E > Target P/E is accretive).\n"
    "   - Debt/cash deals depend strictly on Target Earnings Yield (1 / Target Purchase P/E) vs. After-Tax Cost of Debt [Rate * (1 - Tax)].\n"
    "   - Equity dilution = Deal Consideration / Buyer Market Cap.\n"
    "5. LEVERAGED FINANCE (LEVFIN) RULES:\n"
    "   - Pricing Hierarchy: Subordinated/unsecured debt MUST price HIGHER than senior secured debt (credit risk spread). If TLB is 8.0%, Notes must clear at 8.75%-9.5%.\n"
    "   - Sources & Uses Footing: Percentages MUST divide by Total Sources/Uses and sum to exactly 100.0%.\n"
    "   - High Density Delivery: Keep explanations to 1 concise sentence each so all sections complete before token limits.\n"
    "6. FACTUAL INTEGRITY & MULTI-TURN CONTEXT: Maintain conversational memory across turns. Never fabricate executive biographies, regulatory appointments, or fake personas.\n"
    "7. DCF MODELING: For DCF requests, execute quantitative schedules, render a matplotlib chart, and write 'model.xlsx'."
)

CACHED_WORKING_MODEL = None

def is_quantitative_model_request(prompt: str) -> bool:
    """Accurately distinguishes between requests to build/calculate a DCF model 
    versus qualitative/advisory equity research and M&A queries."""
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
    if has_discount_metric and has_financial_metric:
        return True
        
    return False

def extract_code_from_markdown(text: str) -> str | None:
    m = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m_generic = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if m_generic:
        return m_generic.group(1).strip()
    return None

def sanitize_report_formatting(text: str) -> str:
    cleaned = re.sub(r'```(?:python)?\s*[\s\S]*?```', '', text)
    cleaned = cleaned.replace("** `", "**").replace("` **", "**")
    cleaned = cleaned.replace("`**", "**").replace("**`", "**")
    cleaned = re.sub(r'`([^`\n]+)`', r'\1', cleaned)
    cleaned = cleaned.replace('`', '')
    cleaned = re.sub(r'(?<!\\)\$', r'\$', cleaned)
    return re.sub(r'\n{3,}', '\n\n', cleaned).strip()

def enforce_sandbox_contracts(code_str: str) -> str:
    clean_code = re.sub(r'ExcelWriter\s*\(\s*[\'"][^\'"]+\.xlsx[\'"]', 'ExcelWriter("model.xlsx"', code_str)
    clean_code = re.sub(r'\.to_excel\s*\(\s*[\'"][^\'"]+\.xlsx[\'"]', '.to_excel("model.xlsx"', clean_code)

    preamble = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "model_terms_required = 5\n"
        "years = 5\n"
        "projection_years = 5\n"
        "forecast_period = 5\n"
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
    """Discovers an accessible active model dynamically via live ping tests."""
    global CACHED_WORKING_MODEL
    if CACHED_WORKING_MODEL and not force_refresh and CACHED_WORKING_MODEL != exclude_model:
        return CACHED_WORKING_MODEL

    try:
        models_data = client.models.list().data
        available_ids = [m.id for m in models_data]
    except Exception:
        available_ids = []

    filter_out = [
        "safeguard", "guard", "whisper", "orpheus",
        "allam", "arabic", "vision", "embed", "audio"
    ]
    candidates = [
        mid for mid in available_ids
        if not any(b in mid.lower() for b in filter_out) and mid != exclude_model
    ]

    preference = [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "llama-3.2-3b-preview",
        "llama-3.1-8b-instant"
    ]

    test_queue = [m for m in preference if m in candidates] + [m for m in candidates if m not in preference]
    if not test_queue:
        test_queue = candidates if candidates else preference

    for model_name in test_queue:
        try:
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
            CACHED_WORKING_MODEL = model_name
            return model_name
        except Exception:
            continue

    if candidates:
        CACHED_WORKING_MODEL = candidates[0]
        return CACHED_WORKING_MODEL

    avail_str = ", ".join(available_ids) if available_ids else "None found"
    raise RuntimeError(f"Could not connect to any active Groq text model. Available IDs: [{avail_str}]")

def create_completion_with_fallback(client: Groq, preferred_model: str = None, status_callback=None, **kwargs):
    """Executes completions with auto-throttling, rate limit cooldowns, and cross-model failover."""
    global CACHED_WORKING_MODEL
    
    filter_out = ["guard", "safeguard", "orpheus", "whisper", "allam", "arabic"]
    if preferred_model and not any(b in preferred_model.lower() for b in filter_out):
        active_model = preferred_model
    else:
        active_model = get_best_available_model(client)
        
    if "max_tokens" in kwargs:
        kwargs["max_tokens"] = min(kwargs["max_tokens"], 850)
    else:
        kwargs["max_tokens"] = 850

    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.1

    kwargs["model"] = active_model

    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        err_str = str(e).lower()
        
        # Scenario 1: OTPM Rate Limit Reached
        if "429" in err_str or "rate_limit_exceeded" in err_str or "output tokens per minute" in err_str:
            delay_match = re.search(r'try again in ([0-9.]+)s', err_str)
            sleep_sec = float(delay_match.group(1)) + 0.5 if delay_match else 2.5
            
            if status_callback:
                status_callback("info", f"Rate limit pause ({sleep_sec:.1f}s)... retrying on secondary model.")
            
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

        # Scenario 2: Model decommissioned or access issue
        if "404" in err_str or "model_not_found" in err_str or "decommissioned" in err_str or "400" in err_str or "access" in err_str:
            if status_callback:
                status_callback("info", f"Model `{active_model}` not accessible. Auto-switching...")
            active_model = get_best_available_model(client, exclude_model=active_model, force_refresh=True)
            kwargs["model"] = active_model
            return client.chat.completions.create(**kwargs)
            
        raise e

def build_contextual_messages(user_input: str, system_prompt: str, max_turns: int = 3) -> list[dict]:
    """Retrieves recent conversation turns from Streamlit session state to preserve multi-turn context."""
    history = []
    try:
        import streamlit as st
        if "messages" in st.session_state and isinstance(st.session_state.messages, list):
            candidates = st.session_state.messages[:]
            if candidates and candidates[-1].get("content") == user_input:
                candidates = candidates[:-1]
            
            recent = candidates[-(max_turns * 2):]
            for m in recent:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role in ["user", "assistant"] and content:
                    if role == "assistant" and len(content) > 250:
                        content = content[:250] + "..."
                    history.append({"role": role, "content": str(content)})
    except Exception:
        pass

    return [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_input}]

def parse_prompt_financial_parameters(prompt: str) -> tuple[float, float, float, float, float, int]:
    p = prompt.lower()
    
    rev_match = re.search(r'\$?(\d+(?:\.\d+)?)\s*(?:m|million)?\s*(?:base\s*)?rev', p)
    if not rev_match:
        rev_match = re.search(r'rev(?:enue)?\s*(?:of\s*)?\$?(\d+(?:\.\d+)?)', p)
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
    if include_index:
        headers = ["Parameter"] + [str(col) for col in df.columns]
        rows = [[str(idx)] + [str(val) for val in row] for idx, row in zip(df.index, df.values)]
    else:
        headers = [str(col) for col in df.columns]
        rows = [[str(val) for val in row] for row in df.values]
    
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
            col.append(f"\\${pv_cf + cur_pv_tv:.2f}M")
        sens_data[f"WACC {w*100:.1f}%"] = col

    df_sens = pd.DataFrame(sens_data, index=[f"TG {g*100:.1f}%" for g in tg_range])

    df_proj = pd.DataFrame({
        "Year": [str(y) for y in years_idx],
        "Revenue ($M)": [f"\\${x:.2f}M" for x in rev],
        "EBITDA ($M)": [f"\\${x:.2f}M" for x in ebitda],
        "UFCF ($M)": [f"\\${x:.2f}M" for x in ufcf],
        "Discount Factor": [f"{x:.4f}" for x in df],
        "PV of UFCF ($M)": [f"\\${x:.2f}M" for x in pv_ufcf]
    })

    fig, ax = plt.subplots(figsize=(8, 4.5))
    components = ['PV of Cash Flows', 'PV of Terminal Value']
    values = [sum_pv_ufcf, pv_tv]
    colors = ['#1f77b4', '#2ca02c']
    bars = ax.bar(components, values, color=colors, width=0.45)
    ax.set_title(f'Enterprise Value Breakdown ($M) — Total:${ev:.2f}M', fontsize=12, fontweight='bold', pad=12)
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
        df_summary = pd.DataFrame({
            "Metric": ["PV of 5-Yr Cash Flows ($M)", "Terminal Value ($M)", "PV of Terminal Value ($M)", "Enterprise Value ($M)"],
            "Value": [round(sum_pv_ufcf, 2), round(tv, 2), round(pv_tv, 2), round(ev, 2)]
        })
        df_summary.to_excel(writer, sheet_name='Valuation_Summary', index=False)
        df_sens.to_excel(writer, sheet_name='Sensitivity_Matrix')
    excel_bytes = excel_buf.getvalue()

    summary_df = pd.DataFrame({
        "Metric": [
            "Base Revenue", "Annual Growth", "EBITDA Margin", 
            "WACC (Discount Rate)", "Terminal Growth Rate",
            "PV of 5-Yr Cash Flows", "Terminal Value (Nominal)",
            "PV of Terminal Value", "Enterprise Value"
        ],
        "Value": [
            f"\\${base_rev:.2f}M", f"{growth*100:.1f}%", f"{margin*100:.1f}%",
            f"{wacc*100:.1f}%", f"{tg*100:.2f}%",
            f"\\${sum_pv_ufcf:.2f}M", f"\\${tv:.2f}M",
            f"\\${pv_tv:.2f}M", f"**\\${ev:.2f}M**"
        ]
    })

    proj_table_md = df_to_markdown_table(df_proj, include_index=False)
    summary_table_md = df_to_markdown_table(summary_df, include_index=False)
    sens_table_md = df_to_markdown_table(df_sens, include_index=True)

    calc_briefing = f"""### 📊 Executive Valuation Briefing

An unlevered Discounted Cash Flow (DCF) model was conducted over a **{years}-year discrete forecast horizon** applying a **{wacc*100:.1f}% WACC** discount rate and a **{tg*100:.2f}% perpetual growth rate**.

**Core Valuation Metrics:**
{summary_table_md}

---

### 1. 5-Year Discrete Cash Flow Projections
{proj_table_md}

* **Forecast Cash Flow Expansion:** UFCF expands from **\\${ufcf[0]:.2f}M** (Year 1) to **\\${ufcf[-1]:.2f}M** (Year 5).
* **Discounted Explicit Value:** The 5-year discrete forecast contributes **\\${sum_pv_ufcf:.2f}M** ({(sum_pv_ufcf/ev)*100:.1f}% of Enterprise Value).

---

### 2. Terminal Value & Enterprise Value

* **Gordon Growth Terminal Value:** **\\${tv:.2f}M** (discounted PV of **\\${pv_tv:.2f}M**, representing {(pv_tv/ev)*100:.1f}% of Enterprise Value).
* **Implied Enterprise Value (EV):** **\\${ev:.2f}M**

---

### 3. Sensitivity Analysis (Enterprise Value in $M)
{sens_table_md}
"""

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

    is_dcf_calc = is_quantitative_model_request(user_input)

    # Branch 1: M&A Advisory, FIG Equity Research, LevFin, Restructuring Memos
    if not is_dcf_calc:
        try:
            messages = build_contextual_messages(user_input, SYSTEM_PROMPT, max_turns=3)
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

    # Branch 2: Quantitative DCF Modeling
    generated_plot = None
    generated_excel = None
    structured_briefing = ""

    if status_callback:
        status_callback("info", "Executing quantitative model in sandbox...")

    try:
        messages = build_contextual_messages(user_input, SYSTEM_PROMPT, max_turns=2)
        response = create_completion_with_fallback(
            client=client,
            preferred_model=selected_model,
            status_callback=status_callback,
            messages=messages,
            tools=AGENT_TOOLS,
            tool_choice="auto",
            max_tokens=850,
            temperature=0.1
        )
        msg = response.choices[0].message
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc.function.name == "python_interpreter":
                    args = json.loads(tc.function.arguments or "{}")
                    raw_code = args.get("code", "")
                    if raw_code:
                        safe_code = enforce_sandbox_contracts(raw_code)
                        out, p_bytes, x_bytes = tool_python_interpreter(safe_code, hitl_enabled=hitl_enabled)
                        if p_bytes: generated_plot = p_bytes
                        if x_bytes: generated_excel = x_bytes
    except Exception:
        pass

    # Guaranteed Quantitative Fallback for DCF modeling
    if not generated_plot or not generated_excel:
        if status_callback:
            status_callback("info", "Executing deterministic quantitative engine...")
        briefing_text, plot_bytes, excel_bytes = generate_deterministic_dcf_artifacts(user_input)
        generated_plot = plot_bytes
        generated_excel = excel_bytes
        structured_briefing = briefing_text

    final_text = sanitize_report_formatting(structured_briefing)
    elapsed = time.perf_counter() - start_time
    return final_text, generated_plot, generated_excel, elapsed
