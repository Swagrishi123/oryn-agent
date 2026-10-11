import streamlit as st
import math, re, urllib.parse, requests, io, sys, json, time, base64, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pypdf import PdfReader
from PIL import Image
import yfinance as yf
from groq import Groq
import sympy
from sympy import symbols, Symbol, Eq, solve, simplify, Rational
from duckduckgo_search import DDGS

# --- Page Configuration (Mobile-Optimized) ---
st.set_page_config(
    page_title="ORYN Analyst — Autonomous Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Responsive Modern Purple UI Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff !important;
        font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .oryn-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        padding: 10px 8px 6px 8px;
        border-bottom: 1px solid rgba(74, 20, 140, 0.1);
        text-align: center;
        margin-bottom: 12px;
    }
    .oryn-brand {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    .oryn-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #4a148c !important;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .oryn-subtitle {
        color: #7b1fa2 !important;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 2px 0 0 0;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background-color: #f3e8ff !important;
        border-radius: 16px !important;
        padding: 10px 14px !important;
        border: 1px solid #e9d5ff !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) * {
        color: #6b21a8 !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #faf5ff !important;
        border-radius: 16px !important;
        padding: 12px 16px !important;
        border: 1px solid #f3e8ff !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) *,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) p,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) span {
        color: #4a148c !important;
        font-weight: 500;
        font-size: 0.95rem !important;
    }
    code {
        background-color: #ede9fe !important;
        color: #581c87 !important;
        font-weight: 600 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        word-break: break-word !important;
    }
    .stDownloadButton button, .stButton button {
        min-height: 42px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        border: 1px solid #e9d5ff !important;
        transition: all 0.2s ease !important;
    }
    @media screen and (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 4rem !important;
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
        }
        .oryn-title { font-size: 1.3rem !important; }
        .oryn-subtitle { font-size: 0.7rem !important; }
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        .stDownloadButton button {
            width: 100% !important;
            margin-bottom: 6px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="oryn-header">
    <div class="oryn-brand">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="#4a148c" xmlns="http://www.w3.org/2000/svg">
            <path d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.771-4.2057 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7467-7.0731zM13.2599 22.4042a4.4568 4.4568 0 0 1-2.9001-1.0543l.1422-.0818 4.811-2.778a.791.791 0 0 0 .3955-.6855v-6.786l2.1643 1.25a.1582.1582 0 0 1 .0791.137v5.545a4.4628 4.4628 0 0 1-4.692 4.4536zM3.8205 18.0645a4.4568 4.4568 0 0 1-.5157-3.0568l.1422.0853 4.811 2.778a.791.791 0 0 0 .791 0l5.8753-3.3934v2.5a.1582.1582 0 0 1-.0791.137l-4.8 2.7717a4.4628 4.4628 0 0 1-6.2247-1.8218zm-1.8218-9.4264a4.4568 4.4568 0 0 1 2.3844-2.0025v5.7176a.791.791 0 0 0 .3955.6855l5.8753 3.3934-2.1643 1.25a.1582.1582 0 0 1-.1582 0l-4.8-2.7717a4.4628 4.4628 0 0 1-1.5327-6.2723zm13.931-1.0543-5.8753-3.3934 2.1643-1.25a.1582.1582 0 0 1 .1582 0l4.8 2.7717a4.4628 4.4628 0 0 1-.5157 8.5292v-5.972a.791.791 0 0 0-.7315-.6855zm3.8205 6.3686a4.4568 4.4568 0 0 1 .5157 3.0568l-.1422-.0853-4.811-2.778a.791.791 0 0 0-.791 0l-5.8753 3.3934v-2.5a.1582.1582 0 0 1 .0791-.137l4.8-2.7717a4.4628 4.4628 0 0 1 6.2247 1.8218zM10.7401 13.0847l-2.6105-1.507 2.6105-1.507 2.6105 1.507z"/>
        </svg>
        <h1 class="oryn-title">ORYN ANALYST</h1>
    </div>
    <p class="oryn-subtitle">Multimodal Vision • Institutional Research • Financial Modeling • Working Memory</p>
</div>
""", unsafe_allow_html=True)

# --- State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = st.secrets.get("GROQ_API_KEY", "")

if "working_memory" not in st.session_state:
    st.session_state.working_memory = []

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []

if "image_insights" not in st.session_state:
    st.session_state.image_insights = ""

if "telemetry" not in st.session_state:
    st.session_state.telemetry = {
        "queries": 0,
        "tool_calls": 0,
        "last_latency_s": 0.0,
        "tokens_est": 0,
        "breakdown": {"python": 0, "web": 0, "wiki": 0, "finance": 0, "doc_search": 0}
    }

# Quick Reset Bar
col_hdr_l, col_hdr_r = st.columns([3, 1])
with col_hdr_r:
    if st.button("🗑️ Reset", use_container_width=True, help="Clear session"):
        st.session_state.messages = []
        st.session_state.working_memory = []
        st.session_state.pdf_chunks = []
        st.session_state.image_insights = ""
        st.session_state.telemetry = {
            "queries": 0, "tool_calls": 0, "last_latency_s": 0.0, "tokens_est": 0,
            "breakdown": {"python": 0, "web": 0, "wiki": 0, "finance": 0, "doc_search": 0}
        }
        st.rerun()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Analyst Controls")
    
    key_input = st.text_input(
        "Groq API Key:",
        value=st.session_state.api_key,
        type="password",
        help="Permanent key: Streamlit Cloud -> Settings -> Secrets"
    )
    if key_input:
        st.session_state.api_key = key_input.strip()

    client = Groq(api_key=st.session_state.api_key) if st.session_state.api_key else None

    active_model = st.selectbox(
        "Reasoning Engine:",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0
    )
    hitl_enabled = st.toggle("🛡️ HITL Safety Guardrails", value=True)

    st.markdown("---")
    st.header("🧠 Working Memory")
    if st.session_state.working_memory:
        for idx, item in enumerate(st.session_state.working_memory):
            st.caption(f"• {item}")
        if st.button("Clear Memory", use_container_width=True):
            st.session_state.working_memory = []
            st.rerun()
    else:
        st.caption("No permanent facts or baseline figures stored yet.")

    st.markdown("---")
    st.header("📊 PM Telemetry Dashboard")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.metric("Total Turns", st.session_state.telemetry["queries"])
        st.metric("Tool Invocations", st.session_state.telemetry["tool_calls"])
    with col_t2:
        st.metric("Last Latency", f"{st.session_state.telemetry['last_latency_s']:.2f}s")
        st.metric("Est. Tokens", st.session_state.telemetry["tokens_est"])

    st.caption("Tool Distribution:")
    b = st.session_state.telemetry["breakdown"]
    st.text(f"Python: {b['python']} | Market: {b['finance']} | Docs: {b['doc_search']} | Web: {b['web']}")

# --- Helpers: Code Sanitizer & Markdown to HTML ---
def clean_code_snippet(code: str) -> str:
    cleaned_lines = []
    for line in code.strip().splitlines():
        trimmed = line.strip()
        if trimmed.startswith("```"):
            continue
        if "plt.show()" in trimmed:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

def markdown_to_clean_html(text: str) -> str:
    text = re.sub(r"^### (.*?)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.*?)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.*?)$", r"<h1>\1</h1>", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)
    text = re.sub(r"`(.*?)`", r"<code>\1</code>", text)
    
    lines = text.split("\n")
    in_list = False
    html_lines = []
    for line in lines:
        line_strip = line.strip()
        if line_strip.startswith("- ") or line_strip.startswith("* "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{line_strip[2:]}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            if line_strip:
                if not line_strip.startswith("<h") and not line_strip.startswith("</"):
                    html_lines.append(f"<p>{line_strip}</p>")
            else:
                html_lines.append("")
    if in_list:
        html_lines.append("</ul>")
    return "\n".join(html_lines)

def build_html_report(text: str, plot_bytes: bytes | None) -> str:
    current_date = datetime.datetime.now().strftime("%B %d, %Y • %H:%M UTC")
    img_html = ""
    if plot_bytes:
        b64 = base64.b64encode(plot_bytes).decode('utf-8')
        img_html = f"""
        <div style="background:#faf5ff; border:1px solid #e9d5ff; border-radius:12px; padding:20px; margin:24px 0; text-align:center;">
            <div style="font-size:11px; font-weight:700; color:#7b1fa2; margin-bottom:10px; text-align:left;">VISUAL ARTIFACT</div>
            <img src="data:image/png;base64,{b64}" style="max-width:100%; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.06);"/>
        </div>
        """
    body_html = markdown_to_clean_html(text)
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ORYN Analyst Briefing</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #1e1b4b; padding: 30px 15px; margin: 0; }}
        .card {{ max-width: 820px; margin: 0 auto; background: #ffffff; border-radius: 14px; border: 1px solid #e2e8f0; box-shadow: 0 10px 30px rgba(0,0,0,0.05); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #4a148c, #6a1b9a); color: #fff; padding: 32px 36px; }}
        .header h1 {{ margin: 0 0 6px 0; font-size: 24px; }}
        .content {{ padding: 36px; line-height: 1.6; font-size: 15px; }}
        h2 {{ color: #4a148c; font-size: 19px; border-bottom: 1px solid #f3e8ff; padding-bottom: 6px; margin-top: 24px; }}
        code {{ background: #f3e8ff; color: #581c87; padding: 2px 6px; border-radius: 4px; font-size: 13px; }}
        @media print {{ body {{ padding: 0; background: #fff; }} .card {{ border: none; box-shadow: none; }} }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <h1>ORYN Analyst — Executive Briefing</h1>
            <div style="font-size:13px; opacity:0.85;">Generated autonomously • {current_date}</div>
        </div>
        <div class="content">{img_html}{body_html}</div>
    </div>
</body>
</html>"""

# --- Python Sandbox with Excel & Matplotlib Interceptors ---
def tool_python_interpreter(code: str) -> tuple[str, bytes | None, bytes | None]:
    code = clean_code_snippet(code)
    if hitl_enabled:
        forbidden = ["import os", "import sys", "import subprocess", "import shutil", "__import__", "open("]
        for term in forbidden:
            if term in code:
                return f"🛡️ HITL Intercept: '{term}' restricted.", None, None

    stdout_trap = io.StringIO()
    sys_stdout_backup = sys.stdout
    plot_bytes = None
    excel_bytes = None
    
    scope = {
        "math": math, "sympy": sympy, "symbols": symbols, "Symbol": Symbol,
        "Eq": Eq, "solve": solve, "simplify": simplify, "Rational": Rational,
        "pi": math.pi, "plt": plt, "np": np, "pd": pd, "yf": yf, "__builtins__": __builtins__
    }

    try:
        plt.close('all')
        sys.stdout = stdout_trap
        exec("import math\nimport sympy\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n" + code, scope)
        
        # Capture Matplotlib Plot
        if plt.get_fignums():
            fig_buf = io.BytesIO()
            plt.gcf().savefig(fig_buf, format="png", bbox_inches="tight", dpi=140)
            fig_buf.seek(0)
            plot_bytes = fig_buf.getvalue()
            plt.close('all')

        # Capture DataFrame to Excel
        for var_name, var_val in scope.items():
            if isinstance(var_val, pd.DataFrame) and not var_name.startswith("_"):
                x_buf = io.BytesIO()
                with pd.ExcelWriter(x_buf, engine='openpyxl') as writer:
                    var_val.to_excel(writer, index=True, sheet_name="Data_Model")
                x_buf.seek(0)
                excel_bytes = x_buf.getvalue()
                break

        output = stdout_trap.getvalue().strip()
        return (output if output else "Calculations executed successfully."), plot_bytes, excel_bytes
    except Exception as e:
        plt.close('all')
        return f"Python Execution Error: {e}", None, None
    finally:
        sys.stdout = sys_stdout_backup

# --- Domain Tools: Financial, Web, Document Search ---
def tool_get_financial_data(ticker: str, data_type: str = "summary") -> str:
    try:
        t = yf.Ticker(ticker.strip().upper())
        if data_type == "income_stmt":
            df = t.quarterly_income_stmt
            return f"{ticker} Quarterly Income Statement (top 5 rows):\n" + df.head(5).to_string()
        elif data_type == "balance_sheet":
            df = t.quarterly_balance_sheet
            return f"{ticker} Quarterly Balance Sheet (top 5 rows):\n" + df.head(5).to_string()
        else:
            info = t.info
            metrics = {
                "Company": info.get("shortName", ticker),
                "Current Price": info.get("currentPrice"),
                "Market Cap": info.get("marketCap"),
                "Forward P/E": info.get("forwardPE"),
                "Trailing P/E": info.get("trailingPE"),
                "Profit Margin": info.get("profitMargins"),
                "52W High": info.get("fiftyTwoWeekHigh"),
                "52W Low": info.get("fiftyTwoWeekLow")
            }
            return json.dumps(metrics, indent=2)
    except Exception as e:
        return f"Market Data Error for {ticker}: {e}"

def tool_search_document(query: str) -> str:
    if not st.session_state.pdf_chunks:
        return "No PDF document uploaded or indexed."
    q_words = set(query.lower().split())
    scored = []
    for c in st.session_state.pdf_chunks:
        score = sum(1 for w in q_words if w in c.lower())
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        top_matches = [f"Excerpt {i+1}:\n{item[1]}" for i, item in enumerate(scored[:2])]
        return "\n\n---\n\n".join(top_matches)
    return "No matching sections found in document for query."

def tool_web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results: return f"No web results for '{query}'."
        return "\n".join([f"• [{r.get('title', '')}]: {r.get('body', '')[:280]}" for r in results])
    except Exception as e: return f"Search Error: {e}"

def tool_wikipedia_search(query: str) -> str:
    try:
        url = f"[https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=](https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=){urllib.parse.quote(query)}&utf8=&format=json"
        res = requests.get(url, headers={"User-Agent": "ORYN-Analyst/3.0"}, timeout=4).json()
        hits = res.get("query", {}).get("search", [])
        if hits:
            clean = re.sub(r"<[^>]+>", "", hits[0].get("snippet", ""))
            return f"Wikipedia [{hits[0].get('title')}]: {clean}"
        return f"No Wikipedia entries for '{query}'."
    except Exception as e: return f"Wiki Error: {e}"

def tool_update_working_memory(fact: str) -> str:
    st.session_state.working_memory.append(fact.strip())
    return f"Stored into working memory: '{fact}'"

# --- Tool Schemas ---
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": "Runs Python (math, numpy, pandas, matplotlib, sympy). To create a downloadable Excel model, assign results to a DataFrame named `df`.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Executable Python code."}},
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_data",
            "description": "Fetches real stock prices, valuation multiples, and SEC financial statements for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL, TSLA, NVDA)."},
                    "data_type": {"type": "string", "enum": ["summary", "income_stmt", "balance_sheet"], "description": "Data type."}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": "Searches through uploaded PDF research reports, 10-K filings, and investor memos.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Specific search keyword or topic."}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_working_memory",
            "description": "Saves an established fact, assumption, or baseline metric across conversation turns.",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string", "description": "Fact or metric to store."}},
                "required": ["fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the live web for recent news, market competitors, and updates.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query."}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Searches Wikipedia for encyclopedic background definitions.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search term."}},
                "required": ["query"]
            }
        }
    }
]

# --- Multimodal Upload Ingestion Drawer ---
with st.expander("📎 Multimodal Data Room (PDF / Images)", expanded=False):
    tab_doc, tab_img = st.tabs(["📄 Financial Report (PDF)", "🖼️ Visual Parsing (Images)"])
    
    with tab_doc:
        uploaded_pdf = st.file_uploader("Upload PDF Brief / 10-K", type=["pdf"], key="up_pdf")
        if uploaded_pdf and not st.session_state.pdf_chunks:
            reader = PdfReader(uploaded_pdf)
            full_chunks = []
            for i, p in enumerate(reader.pages):
                text = p.extract_text() or ""
                if len(text.strip()) > 50:
                    full_chunks.append(f"[Page {i+1}] {text.strip()}")
            st.session_state.pdf_chunks = full_chunks
            st.success(f"Indexed {len(full_chunks)} pages for deep search.")

    with tab_img:
        uploaded_img = st.file_uploader("Upload Chart / Table / Screenshot", type=["png", "jpg", "jpeg"], key="up_img")
        if uploaded_img and not st.session_state.image_insights:
            if client:
                with st.spinner("🧠 Vision Model inspecting image..."):
                    try:
                        img_bytes = uploaded_img.getvalue()
                        b64_img = base64.b64encode(img_bytes).decode('utf-8')
                        vision_resp = client.chat.completions.create(
                            model="llama-3.2-11b-vision-preview",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "Extract all numerical data, tables, chart trendlines, and metrics from this image with high precision."},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                                    ]
                                }
                            ]
                        )
                        st.session_state.image_insights = vision_resp.choices[0].message.content
                        st.success("Image parsed into Agent Memory.")
                    except Exception as err:
                        st.error(f"Vision Parsing error: {err}")
            else:
                st.warning("Provide Groq API key first.")

    if st.session_state.image_insights:
        st.info(f"**Parsed Image Context:**\n{st.session_state.image_insights[:300]}...")

# --- Render Chat History ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image"):
