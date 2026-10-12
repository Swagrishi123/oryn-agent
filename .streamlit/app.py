import streamlit as st
import io, re, base64, datetime
from agent import run_analyst_agent

# --- Page Configuration ---
st.set_page_config(
    page_title="ORYN Analyst — Autonomous Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Responsive Purple Theme Styling ---
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
    <p class="oryn-subtitle">Institutional BFSI Research • Financial Modeling • Quantitative Analysis</p>
</div>
""", unsafe_allow_html=True)

# --- State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = st.secrets.get("GROQ_API_KEY", "")

# Header Reset Button
col_hdr_l, col_hdr_r = st.columns([3, 1])
with col_hdr_r:
    if st.button("🗑️ Reset", use_container_width=True, help="Clear session"):
        st.session_state.messages = []
        st.rerun()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Analyst Controls")
    key_input = st.text_input(
        "Groq API Key:",
        value=st.session_state.api_key,
        type="password",
        help="API Key for Groq execution"
    )
    if key_input:
        st.session_state.api_key = key_input.strip()

    active_model = st.selectbox(
        "Reasoning Engine:",
        options=["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        index=0,
        help="Use 8b-instant for fast, rate-limit-free execution on free tiers."
    )
    hitl_enabled = st.toggle("🛡️ HITL Safety Guardrails", value=True)

# --- Helpers: HTML Builder & Text Sanitizer ---
def strip_leaked_code(text: str) -> str:
    raw_script_pattern = r"(?:import\s+(?:matplotlib|pandas|numpy|plt|yf)[\s\S]*?(?:plt\.show\(\)|plt\.tight_layout\(\)|plt\.close\(\)))"
    text = re.sub(raw_script_pattern, "\n*(Visual chart artifact rendered below)*\n", text)
    code_block_pattern = r"```(?:python)?\s*(?:import\s+(?:matplotlib|plt)|plt\.)[\s\S]*?```"
    text = re.sub(code_block_pattern, "\n*(Visual chart artifact rendered below)*\n", text)
    text = text.replace("</br>", " ").replace("<br>", " ")
    text = re.sub(r"\[Image:\s*[^\]]+\]", "", text)
    text = re.sub(r"\*\s+\*", "", text)
    text = text.replace(r"\approx", "≈").replace(r"\times", "×")
    return text.strip()

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
    clean_text = strip_leaked_code(text)
    paragraphs = "".join([f"<p>{p.strip()}</p>" for p in clean_text.split("\n\n") if p.strip()])
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ORYN Analyst Briefing</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #1e1b4b; padding: 30px 15px; margin: 0; }}
        .card {{ max-width: 820px; margin: 0 auto; background: #ffffff; border-radius: 14px; border: 1px solid #e2e8f0; box-shadow: 0 10px 30px rgba(0,0,0,0.05); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #4a148c, #6a1b9a); color: #fff; padding: 32px 36px; }}
        .header h1 {{ margin: 0 0 6px 0; font-size: 24px; }}
        .content {{ padding: 36px; line-height: 1.6; font-size: 15px; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <h1>ORYN Analyst — Executive Briefing</h1>
            <div style="font-size:13px; opacity:0.85;">Generated autonomously • {current_date}</div>
        </div>
        <div class="content">{img_html}{paragraphs}</div>
    </div>
</body>
</html>"""

# --- Render Chat History ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg.get("role", "assistant")):
        st.markdown(msg.get("content", ""))
        if msg.get("image"):
            st.image(msg["image"], caption="Data Visualization", use_container_width=True)
            
        c_dl = st.columns(3)
        if msg.get("image"):
            with c_dl[0]:
                st.download_button("📊 Download Chart (.png)", msg["image"], f"chart_{idx}.png", "image/png", key=f"dl_p_{idx}", use_container_width=True)
            with c_dl[1]:
                html_doc = build_html_report(msg.get("content", ""), msg["image"])
                st.download_button("📑 Download Report (.html)", html_doc, f"report_{idx}.html", "text/html", key=f"dl_h_{idx}", use_container_width=True)
        if msg.get("excel"):
            with c_dl[2]:
                st.download_button("📗 Financial Model (.xlsx)", msg["excel"], f"model_{idx}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_x_{idx}", use_container_width=True)

# --- User Input & Agent Orchestration ---
user_input = st.chat_input("Ask ORYN Analyst...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not st.session_state.api_key:
        with st.chat_message("assistant"):
            st.error("Please provide a valid Groq API Key in the sidebar.")
    else:
        with st.chat_message("assistant"):
            status_container = st.status("🧠 ORYN Analyst is reasoning...", expanded=True)

            def status_callback(level: str, text: str):
                if level == "warning":
                    status_container.warning(text)
                else:
                    status_container.write(text)

            final_text, generated_plot, generated_excel, elapsed = run_analyst_agent(
                user_input=user_input,
                api_key=st.session_state.api_key,
                selected_model=active_model,
                hitl_enabled=hitl_enabled,
                status_callback=status_callback
            )

            final_text = strip_leaked_code(final_text)
            status_container.update(label=f"✅ Complete ({elapsed:.2f}s)", state="complete", expanded=False)

            st.markdown(final_text)
            if generated_plot:
                st.image(generated_plot, caption="Generated Visual Artifact", use_container_width=True)

            cd = st.columns(3)
            msg_count = len(st.session_state.messages)
            if generated_plot:
                with cd[0]:
                    st.download_button("📊 Download Chart (.png)", generated_plot, "oryn_chart.png", "image/png", key=f"live_p_{msg_count}", use_container_width=True)
                with cd[1]:
                    html_report = build_html_report(final_text, generated_plot)
                    st.download_button("📑 Download Report (.html)", html_report, "oryn_report.html", "text/html", key=f"live_h_{msg_count}", use_container_width=True)
            if generated_excel:
                with cd[2]:
                    st.download_button("📗 Financial Model (.xlsx)", generated_excel, "oryn_model.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"live_x_{msg_count}", use_container_width=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_text,
                "image": generated_plot,
                "excel": generated_excel
            })
