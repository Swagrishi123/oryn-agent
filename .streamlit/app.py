import streamlit as st
import io, re, base64, datetime
import importlib, sys

# Force Streamlit Cloud to reload agent.py from disk instead of RAM
if "agent" in sys.modules:
    importlib.reload(sys.modules["agent"])
import agent
importlib.reload(agent)
from agent import run_analyst_agent

# --- Page Configuration ---
st.set_page_config(
    page_title="ORYN Analyst — Autonomous Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Safe Secrets Retrieval ---
default_api_key = ""
try:
    if "GROQ_API_KEY" in st.secrets:
        default_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    default_api_key = ""

# --- State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = default_api_key

# --- Clean Native Header ---
col_title, col_reset = st.columns([5, 1])
with col_title:
    st.markdown("<h1 style='color: #4a148c; margin-bottom: 0;'>✦ ORYN ANALYST</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #7b1fa2; font-weight: 500; margin-top: 0;'>Institutional BFSI Research • Financial Modeling • Quantitative Analysis</p>", unsafe_allow_html=True)
with col_reset:
    st.write("")
    if st.button("🗑️ Reset", use_container_width=True, help="Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.divider()

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
        options=["llama-3.1-8b-instant"],
        index=0,
        help="Ultra-fast, stable institutional engine"
    )
    hitl_enabled = st.toggle("🛡️ HITL Safety Guardrails", value=True)

# --- Helper Functions ---
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
            <img src="data:image/png;base64,{b64}" style="max-width:100%; border-radius:8px;"/>
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
        .card {{ max-width: 820px; margin: 0 auto; background: #ffffff; border-radius: 14px; border: 1px solid #e2e8f0; padding: 36px; }}
        h1 {{ color: #4a148c; margin-top: 0; }}
        .content {{ line-height: 1.6; font-size: 15px; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>ORYN Analyst — Executive Briefing</h1>
        <div style="font-size:13px; color:#6b7280; margin-bottom: 24px;">Generated autonomously • {current_date}</div>
        <div class="content">{img_html}{paragraphs}</div>
    </div>
</body>
</html>"""

# --- Render Chat History ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg.get("role", "assistant")):
        st.markdown(msg.get("content", ""))
        if msg.get("image"):
            st.image(msg["image"], caption="Generated Visual Artifact", use_container_width=True)
            
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
            st.error("Please enter your Groq API Key in the left sidebar to proceed.")
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
