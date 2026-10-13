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
    "You are ORYN Analyst, an institutional BFSI quantitative analyst.\n"
    "CORE RULES:\n"
    "1. Respond naturally in the language used by the user.\n"
    "2. For general or conversational questions (e.g., 'What is the capital of China?'), "
    "answer directly and concisely. Do NOT invoke valuation models or tools.\n"
    "3. For valuation and DCF requests, ALWAYS execute python_interpreter to calculate cash flows, "
    "create a matplotlib chart, and write 'model.xlsx'.\n"
    "4. Output clean Markdown tables and executive summaries.\n"
    "   - NEVER use backticks (`) around currency, numbers, or phrases.\n"
    "   - Format currency values cleanly as '$346M' or '346 million USD'."
)

VALUATION_KEYWORDS = [
    "dcf", "valuation", "wacc", "discount rate", "ebitda",
    "terminal growth", "sensitivity matrix", "financial model",
    "cash flow", "excel", "lbo", "irr", "npv", "unlevered"
]

CACHED_WORKING_MODEL = None

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

def get_best_available_model(client: Groq) -> str:
    global CACHED_WORKING_MODEL
    if CACHED_WORKING_MODEL:
        return CACHED_WORKING_MODEL

    try:
        all_ids = [m.id for m in client.models.list().data]
    except Exception:
        all_ids = []

    valid_candidates = [
        mid for mid in all_ids
        if "/" not in mid and not any(b in mid.lower() for b in ["guard", "whisper", "vision", "embed", "audio", "orpheus", "allam", "arabic"])
    ]

    priority = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama3-70b-8192",
        "llama-3.1-8b-instant",
        "llama-3.2-3b-preview",
        "deepseek-r1-distill-llama-70b",
        "mixtral-8x7b-32768"
    ]

    ordered = [p for p in priority if p in valid_candidates] + [c for c in valid_candidates if c not in priority]

    for cand in ordered:
        try:
            client.chat.completions.create(
                model=cand,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
            CACHED_WORKING_MODEL = cand
            return cand
        except Exception:
            continue

    CACHED_WORKING_MODEL = "llama-3.1-8b-instant"
    return CACHED_WORKING_MODEL

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
    """Pure Python table formatter that guarantees clean Markdown without external dependencies."""
    if include_index:
        headers = [""] + [str(col) for col in df.columns]
        rows = [[str(idx)] + [str(val) for val in row] for idx, row in zip(df.index, df.values)]
    else:
        headers = [str(col) for col in df.columns]
        rows = [[str(val) for val in row] for row in df.values]
    
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header_line, sep_line] + row_lines)

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
            col.append(f"{pv_cf + cur_pv_tv:.2f}")
        sens_data[f"WACC {w*100:.1f}%"] = col

    df_sens = pd.DataFrame(sens_data, index=[f"TG {g*100:.1f}%" for g in tg_range])

    df_proj = pd.DataFrame({
        "Year": [str(y) for y in years_idx],
        "Revenue ($M)": [f"{x:.2f}" for x in rev],
        "EBITDA ($M)": [f"{x:.2f}" for x in ebitda],
        "UFCF ($M)": [f"{x:.2f}" for x in ufcf],
        "Discount Factor": [f"{x:.4f}" for x in df],
        "PV of UFCF ($M)": [f"{x:.2f}" for x in pv_ufcf]
    })

    # Render Visual Chart Artifact
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

    # Generate Multi-Sheet Excel Workbook
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

    # Format Structured Institutional Markdown Briefing
    summary_df = pd.DataFrame({
        "Metric": [
            "Base Revenue", "Annual Growth", "EBITDA Margin", 
            "WACC (Discount Rate)", "Terminal Growth Rate",
            "PV of 5-Yr Cash Flows", "Terminal Value (Nominal)",
            "PV of Terminal Value", "Enterprise Value"
        ],
        "Value": [
            f"${base_rev:.2f}M", f"{growth*100:.1f}%", f"{margin*100:.1f}%",
            f"{wacc*100:.1f}%", f"{tg*100:.2f}%",
            f"${sum_pv_ufcf:.2f}M", f"${tv:.2f}M",
            f"${pv_tv:.2f}M", f"**${ev:.2f}M**"
        ]
    })

    proj_table_md = df_to_markdown_table(df_proj, include_index=False)
    summary_table_md = df_to_markdown_table(summary_df, include_index=False)
    sens_table_md = df_to_markdown_table(df_sens, include_index=True)

    calc_briefing = f"""### 📊 Executive Valuation Briefing

An unlevered Discounted Cash Flow (DCF) model was performed over a **{years}-year forecast horizon** applying a **{wacc*100:.1f}% WACC** discount rate and a **{tg*100:.2f}% perpetual growth rate**.

**Core Valuation Metrics:**
{summary_table_md}

---

### 1. 5-Year Discrete Cash Flow Projections

{proj_table_md}

* **Forecast Cash Flow Expansion:** UFCF expands from **${float(df_proj.iloc[0]['UFCF ($M)']):.2f}M** (Year 1) to **${float(df_proj.iloc[-1]['UFCF ($M)']):.2f}M** (Year 5).
* **Discounted Explicit Value:** The 5-year discrete forecast contributes **${sum_pv_ufcf:.2f}M** ({(sum_pv_ufcf/ev)*100:.1f}% of Enterprise Value).

---

### 2. Terminal Value & Enterprise Value

* **Gordon Growth Terminal Value:** **${tv:.2f}M** (discounted PV of **${pv_tv:.2f}M**, representing {(pv_tv/ev)*100:.1f}% of Enterprise Value).
* **Implied Enterprise Value (EV):** **${ev:.2f}M**

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

    active_model = get_best_available_model(client)
    if status_callback:
        status_callback("info", f"Connected to Groq Engine: `{active_model}`")

    is_valuation_query = any(k in user_input.lower() for k in VALUATION_KEYWORDS)

    # 1. Direct answering for non-valuation queries (e.g., "Who is the father of India?")
    if not is_valuation_query:
        try:
            resp = client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=1000
            )
            elapsed = time.perf_counter() - start_time
            return sanitize_report_formatting(resp.choices[0].message.content or ""), None, None, elapsed
        except Exception as e:
            return f"Error: {e}", None, None, time.perf_counter() - start_time

    # 2. Valuation pipeline
    generated_plot = None
    generated_excel = None
    structured_briefing = ""

    if status_callback:
        status_callback("info", "Executing quantitative model in sandbox...")

    try:
        response = client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            tools=AGENT_TOOLS,
            tool_choice="auto",
            max_tokens=2048
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

    # 3. Guaranteed Quantitative Fallback: generates complete Markdown tables, plot, and Excel
    if not generated_plot or not generated_excel:
        if status_callback:
            status_callback("info", "Executing deterministic quantitative engine...")
        briefing_text, plot_bytes, excel_bytes = generate_deterministic_dcf_artifacts(user_input)
        generated_plot = plot_bytes
        generated_excel = excel_bytes
        structured_briefing = briefing_text

    # 4. Final Executive Delivery
    final_text = sanitize_report_formatting(structured_briefing)
    elapsed = time.perf_counter() - start_time
    return final_text, generated_plot, generated_excel, elapsed
