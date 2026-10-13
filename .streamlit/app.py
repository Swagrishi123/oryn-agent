import streamlit as st
import io
import re
import base64
import datetime
import importlib
import sys

# Ensure clean reload of both tools and agent from disk
for mod_name in ["tools", "agent"]:
    if mod_name in sys.modules:
        try:
            importlib.reload(sys.modules[mod_name])
        except Exception:
            pass

import agent
from agent import run_analyst_agent, sanitize_report_formatting

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

# --- Header ---
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
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192"
        ],
        index=0,
        help="Institutional reasoning engines on Groq"
    )
    hitl_enabled = st.toggle("🛡️ HITL Safety Guardrails", value=True)

# --- Canonical UI Cleaner ---
def clean_for_ui(text: str) -> str:
    raw_script_pattern = r"(?:import\s+(?:matplotlib|pandas|numpy|plt|yf)[\s\S]*?(?:plt\.show\(\)|plt\.tight_layout\(\)|plt\.close\(\)))"
    text = re.sub(raw_script_pattern, "\n*(Visual chart artifact rendered below)*\n", text)
    code_block_pattern = r"```(?:python)?\s*(?:import\s+(?:matplotlib|plt)|plt\.)[\s\S]*?```"
    text = re.sub(code_block_pattern, "\n*(Visual chart artifact rendered below)*\n", text)
    text = text.replace("</br>", " ").replace("<br>", " ")
    text = re.sub(r"\[Image:\s*[^\]]+\]", "", text)
    text = re.sub(r"\*\s+\*", "", text)
    text = text.replace(r"\approx", "≈").replace(r"\times", "×")
    return sanitize_report_formatting(text)

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
    clean_text = clean_for_ui(text)
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

def render_download_suite(text: str, plot_bytes: bytes | None, excel_bytes: bytes | None, key_prefix: str):
    buttons = [
        {
            "label": "📥 Download Memo (.txt)",
            "data": text,
            "file_name": f"ORYN_Institutional_Memo_{key_prefix}.txt",
            "mime": "text/plain"
        },
        {
            "label": "📑 Download Report (.html)",
            "data": build_html_report(text, plot_bytes),
            "file_name": f"ORYN_Report_{key_prefix}.html",
            "mime": "text/html"
        }
    ]
    if plot_bytes:
        buttons.append({
            "label": "📊 Download Chart (.png)",
            "data": plot_bytes,
            "file_name": f"ORYN_Chart_{key_prefix}.png",
            "mime": "image/png"
        })
    if excel_bytes:
        buttons.append({
            "label": "📗 Financial Model (.xlsx)",
            "data": excel_bytes,
            "file_name": f"ORYN_Model_{key_prefix}.xlsx",
            "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        })

    cols = st.columns(len(buttons))
    for i, btn in enumerate(buttons):
        with cols[i]:
            st.download_button(
                label=btn["label"],
                data=btn["data"],
                file_name=btn["file_name"],
                mime=btn["mime"],
                key=f"dl_{key_prefix}_{i}",
                use_container_width=True
            )

# --- Render Chat History ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg.get("role", "assistant")):
        st.markdown(clean_for_ui(msg.get("content", "")))
        if msg.get("image"):
            st.image(msg["image"], caption="Generated Visual Artifact", use_container_width=True)
        if msg.get("role") == "assistant":
            render_download_suite(
                text=msg.get("content", ""),
                plot_bytes=msg.get("image"),
                excel_bytes=msg.get("excel"),
                key_prefix=f"hist_{idx}"
            )

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

            status_container.update(label=f"✅ Complete ({elapsed:.2f}s)", state="complete", expanded=False)

            clean_display_text = clean_for_ui(final_text)
            st.markdown(clean_display_text)

            if generated_plot:
                st.image(generated_plot, caption="Generated Visual Artifact", use_container_width=True)

            msg_idx = len(st.session_state.messages)
            render_download_suite(
                text=clean_display_text,
                plot_bytes=generated_plot,
                excel_bytes=generated_excel,
                key_prefix=f"live_{msg_idx}"
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": clean_display_text,
                "image": generated_plot,
                "excel": generated_excel
            })
