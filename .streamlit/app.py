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
from google import genai
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

# --- Modern Purple UI Styling ---
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

if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = st.secrets.get("GEMINI_API_KEY", "")

if "working_memory" not in st.session_state:
    st.session_state.working_memory = []

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []

if "image_insights" not in st.session_state:
    st.session_state.image_insights = ""

if "last_image_name" not in st.session_state:
    st.session_state.last_image_name = ""

if "telemetry" not in st.session_state:
    st.session_state.telemetry = {
        "queries": 0,
        "tool_calls": 0,
        "last_latency_s": 0.0,
        "tokens_est": 0,
        "breakdown": {"python": 0, "web": 0, "wiki": 0, "finance": 0, "doc_search": 0}
    }

# Reset Button
col_hdr_l, col_hdr_r = st.columns([3, 1])
with col_hdr_r:
    if st.button("🗑️ Reset", use_container_width=True, help="Clear session"):
        st.session_state.messages = []
        st.session_state.working_memory = []
        st.session_state.pdf_chunks = []
        st.session_state.image_insights = ""
        st.session_state.last_image_name = ""
        st.session_state.telemetry = {
            "queries": 0, "tool_calls": 0, "last_latency_s": 0.0, "tokens_est": 0,
            "breakdown": {"python": 0, "web": 0, "wiki": 0, "finance": 0, "doc_search": 0}
        }
        st.rerun()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Analyst Controls")
    key_input = st.text_input(
        "Groq API Key (Engine):",
        value=st.session_state.api_key,
        type="password",
        help="Groq API Key for fast reasoning & tools"
    )
    if key_input:
        st.session_state.api_key = key_input.strip()

    gemini_input = st.text_input(
        "Gemini API Key (Vision Parser):",
        value=st.session_state.gemini_key,
        type="password",
        help="Free key from Google AI Studio to inspect uploaded charts/images"
    )
    if gemini_input:
        st.session_state.gemini_key = gemini_input.strip()

    client = Groq(api_key=st.session_state.api_key) if st.session_state.api_key else None

    discovered_models = []
    if client:
        try:
            raw_models = [m.id for m in client.models.list().data]
            banned = ["whisper", "guard", "embed", "safeguard", "orpheus", "canopy", "audio", "tts"]
            discovered_models = [m for m in raw_models if not any(b in m.lower() for b in banned)]
        except Exception:
            discovered_models = []

    if not discovered_models:
        discovered_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    active_model = st.selectbox("Reasoning Engine:", options=discovered_models, index=0)
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
        st.caption("No permanent facts stored yet.")

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

# --- Helpers ---
def clean_code_snippet(code: str) -> str:
    cleaned_lines = []
    for line in code.strip().splitlines():
        trimmed = line.strip()
        if trimmed.startswith("```") or "plt.show()" in trimmed:
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

# --- Python Sandbox ---
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
        
        if plt.get_fignums():
            fig_buf = io.BytesIO()
            plt.gcf().savefig(fig_buf, format="png", bbox_inches="tight", dpi=140)
            fig_buf.seek(0)
            plot_bytes = fig_buf.getvalue()
            plt.close('all')

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

# --- Domain Tools ---
def tool_get_financial_data(ticker: str, data_type: str = "summary") -> str:
    try:
        sym = ticker.strip().upper()
        t = yf.Ticker(sym)
        fi = t.fast_info
        price = getattr(fi, 'last_price', None) or getattr(fi, 'previous_close', 'N/A')
        mcap = getattr(fi, 'market_cap', None)
        fifty_high = getattr(fi, 'year_high', 'N/A')
        fifty_low = getattr(fi, 'year_low', 'N/A')
        
        mcap_str = f"${mcap:,.0f}" if isinstance(mcap, (int, float)) else "N/A"
        price_str = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)

        metrics = {
            "Ticker": sym,
            "Current Price": price_str,
            "Market Cap": mcap_str,
            "52W High": fifty_high,
            "52W Low": fifty_low
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

def tool_update_working_memory(fact: str) -> str:
    st.session_state.working_memory.append(fact.strip())
    return f"Stored into working memory: '{fact}'"

# --- Tool Schemas ---
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": "Runs Python (numpy, pandas, matplotlib). Assign results to DataFrame `df` to trigger automatic Excel export.",
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
            "description": "Fetches current stock prices and market caps rapidly.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "Stock ticker (e.g. AAPL, MSFT, TSLA)."}},
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": "Searches through uploaded PDF research reports.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Specific search query."}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_working_memory",
            "description": "Saves an established fact or metric permanently.",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string", "description": "Fact to save."}},
                "required": ["fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the live web for facts.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query."}},
                "required": ["query"]
            }
        }
    }
]

# --- Multimodal Upload Drawer ---
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
        uploaded_img = st.file_uploader("Upload Chart / Screenshot", type=["png", "jpg", "jpeg"], key="up_img")
        if uploaded_img and (not st.session_state.image_insights or st.session_state.last_image_name != uploaded_img.name):
            st.session_state.last_image_name = uploaded_img.name
            img_bytes = uploaded_img.getvalue()

            # Priority 1: High-Precision Gemini Vision Eye (if key provided)
            if st.session_state.gemini_key:
                with st.spinner("🧠 Vision Engine inspecting image..."):
                    try:
                        g_client = genai.Client(api_key=st.session_state.gemini_key)
                        pil_img = Image.open(io.BytesIO(img_bytes))
                        v_res = g_client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=['Extract and describe all numbers, tables, pie chart slices, labels, and trends with high quantitative precision.', pil_img]
                        )
                        st.session_state.image_insights = v_res.text
                        st.success(f"Image `{uploaded_img.name}` parsed successfully into memory.")
                    except Exception as err:
                        st.error(f"Gemini Vision parsing error: {err}")
            # Priority 2: Fallback to Groq if user has custom vision access enabled
            elif client:
                with st.spinner("🧠 Checking Groq vision access..."):
                    b64_img = base64.b64encode(img_bytes).decode('utf-8')
                    groq_success = False
                    for v_mod in ["llama-3.2-11b-vision", "llama-3.2-90b-vision"]:
                        try:
                            vision_resp = client.chat.completions.create(
                                model=v_mod,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": "Extract all numbers, chart slices, and labels with high precision."},
                                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                                        ]
                                    }
                                ]
                            )
                            st.session_state.image_insights = vision_resp.choices[0].message.content
                            st.success(f"Image parsed via Groq `{v_mod}`.")
                            groq_success = True
                            break
                        except Exception:
                            continue
                    if not groq_success:
                        st.warning("⚠️ Groq tier has no vision models. Enter a free Gemini API Key in the sidebar for full chart inspection, or paste values into chat.")
            else:
                st.warning("Please supply an API Key in the sidebar.")

    if st.session_state.image_insights:
        st.info(f"**Parsed Image Context ({st.session_state.last_image_name}):**\n{st.session_state.image_insights[:300]}...")

# --- Render Chat History ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg.get("role", "assistant")):
        st.markdown(msg.get("content", ""))
        if msg.get("image"):
            st.image(msg["image"], caption="Data Visualization", use_container_width=True)
            
        c_dl = st.columns(3)
        has_buttons = False
        if msg.get("image"):
            with c_dl[0]:
                st.download_button("📊 Download Chart (.png)", msg["image"], f"chart_{idx}.png", "image/png", key=f"dl_p_{idx}", use_container_width=True)
            with c_dl[1]:
                html_doc = build_html_report(msg.get("content", ""), msg["image"])
                st.download_button("📑 Download Report (.html)", html_doc, f"report_{idx}.html", "text/html", key=f"dl_h_{idx}", use_container_width=True)
            has_buttons = True
        if msg.get("excel"):
            with c_dl[2]:
                st.download_button("📗 Financial Model (.xlsx)", msg["excel"], f"model_{idx}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_x_{idx}", use_container_width=True)
            has_buttons = True
        if not has_buttons and msg.get("download") and msg.get("content"):
            st.download_button("📥 Download Summary (.txt)", msg["content"], f"memo_{idx}.txt", "text/plain", key=f"dl_t_{idx}", use_container_width=True)

# --- User Input & Autonomous Execution Loop ---
user_input = st.chat_input("Ask ORYN Analyst...")

if user_input:
    start_time = time.perf_counter()
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not client:
        with st.chat_message("assistant"):
            st.error("Please supply a valid Groq API Key.")
    else:
        system_prompt = (
            "You are ORYN Analyst, an autonomous quantitative and market intelligence analyst built by Mriganka.\n"
            "BEHAVIOR RULES:\n"
            "- For general knowledge, direct questions, or simple chat, answer DIRECTLY and concisely without forcing financial sections.\n"
            "- When asked for financial research, comparisons, or modeling, call the relevant tools and deliver a complete executive briefing.\n"
        )

        if st.session_state.last_image_name and not st.session_state.image_insights:
            system_prompt += f"\nNOTICE: The user uploaded an image named '{st.session_state.last_image_name}', but no vision key was provided to parse it. If asked about the picture, explain that vision parsing requires a Gemini API key in the sidebar."

        if st.session_state.working_memory:
            system_prompt += f"\n\nCURRENT WORKING MEMORY:\n" + "\n".join([f"- {m}" for m in st.session_state.working_memory])
        if st.session_state.image_insights:
            system_prompt += f"\n\nPARSED VISUAL ARTIFACTS / OCR DATA ({st.session_state.last_image_name}):\n{st.session_state.image_insights}"

        agent_messages = [{"role": "system", "content": system_prompt}]
        for m in st.session_state.messages[-4:]:
            if m.get("content"):
                agent_messages.append({"role": m["role"], "content": m["content"]})

        with st.chat_message("assistant"):
            status_container = st.status("🧠 ORYN Analyst is reasoning...", expanded=True)
            max_iterations = 6
            iteration = 0
            final_text = ""
            generated_plot = None
            generated_excel = None
            step_tool_count = 0

            candidate_models = [active_model] + [m for m in discovered_models if m != active_model]

            while iteration < max_iterations:
                iteration += 1
                response = None
                
                for candidate in candidate_models:
                    try:
                        response = client.chat.completions.create(
                            model=candidate,
                            messages=agent_messages,
                            tools=AGENT_TOOLS,
                            tool_choice="auto",
                            max_tokens=900
                        )
                        break
                    except Exception:
                        continue

                if not response:
                    final_text = "All model engines failed to process the request."
                    break

                response_message = response.choices[0].message
                agent_messages.append(response_message)

                if getattr(response_message, "tool_calls", None):
                    for tool_call in response_message.tool_calls:
                        step_tool_count += 1
                        fn_name = tool_call.function.name
                        try:
                            fn_args = json.loads(tool_call.function.arguments)
                        except Exception:
                            fn_args = {}

                        status_container.write(f"🛠️ **Tool:** `{fn_name}`")

                        if fn_name == "python_interpreter":
                            c_str = fn_args.get("code", "")
                            status_container.code(clean_code_snippet(c_str), language="python")
                            out, p_bytes, x_bytes = tool_python_interpreter(c_str)
                            if p_bytes:
                                generated_plot = p_bytes
                                status_container.write("📊 *Plot generated successfully.*")
                            if x_bytes:
                                generated_excel = x_bytes
                                status_container.write("📗 *Financial Excel model created.*")
                            tool_result = out
                            st.session_state.telemetry["breakdown"]["python"] += 1

                        elif fn_name == "get_financial_data":
                            tk = fn_args.get("ticker", "")
                            tool_result = tool_get_financial_data(tk)
                            status_container.write(f"📈 *Retrieved data for {tk}.*")
                            st.session_state.telemetry["breakdown"]["finance"] += 1

                        elif fn_name == "search_document":
                            tool_result = tool_search_document(fn_args.get("query", ""))
                            st.session_state.telemetry["breakdown"]["doc_search"] += 1

                        elif fn_name == "update_working_memory":
                            tool_result = tool_update_working_memory(fn_args.get("fact", ""))

                        elif fn_name == "web_search":
                            tool_result = tool_web_search(fn_args.get("query", ""))
                            st.session_state.telemetry["breakdown"]["web"] += 1

                        else:
                            tool_result = f"Unknown tool `{fn_name}`."

                        agent_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": fn_name,
                            "content": tool_result
                        })
                else:
                    final_text = response_message.content or ""
                    break

            if step_tool_count > 0 and (not final_text or len(final_text.strip()) == 0):
                status_container.write("📝 *Synthesizing briefing...*")
                agent_messages.append({
                    "role": "user",
                    "content": "Synthesize a structured executive briefing based on the tool outputs above. Do not call any tools."
                })
                try:
                    synth_resp = client.chat.completions.create(
                        model=active_model,
                        messages=agent_messages,
                        max_tokens=900
                    )
                    final_text = synth_resp.choices[0].message.content or "Analysis completed."
                except Exception as e:
                    final_text = f"Synthesis note: {e}"

            if not generated_plot and ("```python" in final_text or "plt." in final_text):
                status_container.write("⚡ *Executing inline plot script...*")
                code_matches = re.findall(r"```(?:python)?\s*(.*?)\s*```", final_text, re.DOTALL)
                target_code = "\n".join(code_matches) if code_matches else final_text
                out, p_bytes, x_bytes = tool_python_interpreter(target_code)
                if p_bytes: generated_plot = p_bytes
                if x_bytes: generated_excel = x_bytes

            elapsed = time.perf_counter() - start_time
            status_container.update(label=f"✅ Complete ({elapsed:.2f}s)", state="complete", expanded=False)

            st.session_state.telemetry["queries"] += 1
            st.session_state.telemetry["tool_calls"] += step_tool_count
            st.session_state.telemetry["last_latency_s"] = elapsed
            st.session_state.telemetry["tokens_est"] += len(user_input.split()) * 2 + len(final_text.split()) * 2

            st.markdown(final_text)
            if generated_plot:
                st.image(generated_plot, caption="Generated Visual Artifact", use_container_width=True)

            cd = st.columns(3)
            if generated_plot:
                with cd[0]:
                    st.download_button("📊 Download Chart (.png)", generated_plot, "oryn_chart.png", "image/png", key=f"dl_live_png_{len(st.session_state.messages)}", use_container_width=True)
                with cd[1]:
                    html_report = build_html_report(final_text, generated_plot)
                    st.download_button("📑 Download Report (.html)", html_report, "oryn_executive_report.html", "text/html", key=f"dl_live_html_{len(st.session_state.messages)}", use_container_width=True)
            if generated_excel:
                with cd[2]:
                    st.download_button("📗 Financial Model (.xlsx)", generated_excel, "oryn_model.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_live_xlsx_{len(st.session_state.messages)}", use_container_width=True)
            elif not generated_plot and not generated_excel:
                st.download_button("📥 Download Summary (.txt)", final_text, "oryn_report.txt", "text/plain", key=f"dl_live_txt_{len(st.session_state.messages)}", use_container_width=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_text,
                "image": generated_plot,
                "excel": generated_excel,
                "download": True
            })
