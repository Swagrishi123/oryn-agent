"""ORYN.AI Streamlit Interface (Presentation Layer Only)."""
import os
import streamlit as st
from oryn.agent import OrynAgent

st.set_page_config(
    page_title="ORYN.AI — Autonomous Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Safe Secrets Retrieval
default_api_key = ""
try:
    if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
        default_api_key = str(st.secrets["GROQ_API_KEY"]).strip()
except Exception:
    pass
if not default_api_key:
    default_api_key = os.environ.get("GROQ_API_KEY", "").strip()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header
col1, col2 = st.columns([5, 1])
with col1:
    st.markdown("<h1 style='color: #4a148c; margin-bottom: 0;'>✦ ORYN.AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #7b1fa2; margin-top: 0;'>Modular Personal Agent • Research • Financial Modeling • Safe Execution</p>", unsafe_allow_html=True)
with col2:
    if st.button("🗑️ Reset", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.divider()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Groq API Key:",
        value=default_api_key,
        type="password",
        help="Groq API key for autonomous reasoning"
    )
    model = st.selectbox(
        "Reasoning Engine:",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    )
    hitl_enabled = st.toggle("🛡️ HITL Safety Guardrails", value=True)

# Render Chat History
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if "plot_bytes" in msg and msg["plot_bytes"]:
            st.image(msg["plot_bytes"], caption="Generated Visual Artifact", use_container_width=True)
        if "excel_bytes" in msg and msg["excel_bytes"]:
            st.download_button(
                label="📗 Download Model (.xlsx)",
                data=msg["excel_bytes"],
                file_name=f"ORYN_Model_{idx}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_{idx}"
            )

# Chat Input & Orchestration
user_input = st.chat_input("Ask ORYN.AI...")
if user_input:
    active_key = api_key.strip()
    if not active_key:
        st.error("Please supply a valid Groq API Key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            agent = OrynAgent(api_key=active_key, model_name=model, hitl_enabled=hitl_enabled)
            with st.spinner("Reasoning and executing verified tools..."):
                response = agent.process(user_input)

            st.markdown(response.content, unsafe_allow_html=True)
            if response.plot_bytes:
                st.image(response.plot_bytes, caption="Generated Visual Artifact", use_container_width=True)
            if response.excel_bytes:
                st.download_button(
                    label="📗 Download Model (.xlsx)",
                    data=response.excel_bytes,
                    file_name="ORYN_Model.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="live_dl"
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.content,
                "plot_bytes": response.plot_bytes,
                "excel_bytes": response.excel_bytes
            })
