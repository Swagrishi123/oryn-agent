import streamlit as st
import sys

# Ensure .streamlit modules are importable
sys.path.append(".streamlit")
import agent

st.set_page_config(
    page_title="ORYN Analyst | Institutional BFSI Quantitative Agent",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Gemini-style rotating '+' attachment button
st.markdown(
    """
    <style>
    /* Styling the attachment '+' button */
    div[data-testid="stButton"] button.attach-btn {
        border-radius: 50% !important;
        width: 44px !important;
        height: 44px !important;
        font-size: 22px !important;
        font-weight: bold !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: 1px solid #4a148c !important;
        background-color: #faf5ff !important;
        color: #4a148c !important;
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.2s ease !important;
        padding: 0 !important;
        line-height: 1 !important;
    }
    div[data-testid="stButton"] button.attach-btn:hover {
        background-color: #f3e8ff !important;
        border-color: #7b1fa2 !important;
    }
    /* Rotate 45 degrees into an 'X' when active */
    .btn-rotated button {
        transform: rotate(45deg) !important;
        background-color: #4a148c !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_vault" not in st.session_state:
    st.session_state.session_vault = []

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = None

# ==========================================
# SIDEBAR (RESET & SYSTEM CONFIG ONLY)
# ==========================================
with st.sidebar:
    st.title("🏛️ ORYN Analyst")
    st.caption("Institutional BFSI Quantitative Intelligence")
    st.markdown("---")

    # API Key & Engine Config
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=st.secrets.get("GROQ_API_KEY", ""),
        help="Enter your Groq API key (starts with gsk_)"
    )

    selected_model = st.selectbox(
        "Inference Engine",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0
    )

    st.markdown("---")

    # Reset Controls
    st.subheader("🔄 Session Control")
    vault_count = len(st.session_state.session_vault)
    st.caption(f"Memory Vault: **{vault_count}/3 sessions** preserved across resets.")

    if st.button("🗑️ Reset Chat Screen", use_container_width=True):
        if st.session_state.messages:
            st.session_state.session_vault.append(list(st.session_state.messages))
            if len(st.session_state.session_vault) > 3:
                st.session_state.session_vault.pop(0)
        st.session_state.messages = []
        st.session_state.active_pdf = None
        st.session_state.show_uploader = False
        st.rerun()

    if st.session_state.session_vault:
        if st.button("Purge All Vault Memory", use_container_width=True):
            st.session_state.session_vault = []
            st.session_state.messages = []
            st.session_state.active_pdf = None
            st.session_state.show_uploader = False
            st.rerun()

# ==========================================
# MAIN INTERACTION PANEL
# ==========================================
st.markdown("### Institutional Quantitative Workspace")

# Render active chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("plot_bytes"):
            st.image(message["plot_bytes"], caption="Quantitative Artifact")
        if message.get("excel_bytes"):
            st.download_button(
                label="📥 Download model.xlsx",
                data=message["excel_bytes"],
                file_name="model.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_{message.get('timestamp', 0)}"
            )

# ==========================================
# GEMINI-STYLE ATTACHMENT TRAY & CHAT BAR
# ==========================================
col_btn, col_badge = st.columns([1, 15])

with col_btn:
    btn_active_class = "btn-rotated" if st.session_state.show_uploader else ""
    st.markdown(f'<div class="{btn_active_class}">', unsafe_allow_html=True)
    if st.button("➕", help="Attach 10-K, Credit Agreement, or CIM (up to 200MB)"):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col_badge:
    if st.session_state.active_pdf is not None:
        c1, c2 = st.columns([4, 1])
        with c1:
            st.info(f"📄 **Attached Filing:** {st.session_state.active_pdf.name} ({(st.session_state.active_pdf.size)/(1024*1024):.1f} MB)")
        with c2:
            if st.button("Remove ✕", help="Detach this document"):
                st.session_state.active_pdf = None
                st.rerun()

# Expandable Document Reader Tray
if st.session_state.show_uploader:
    with st.container(border=True):
        st.markdown("**📑 Upload Institutional Filing (up to 200MB)**")
        uploaded_file = st.file_uploader(
            "Upload 10-K, CIM, or Indenture",
            type=["pdf"],
            label_visibility="collapsed",
            help="Upload dense institutional filings up to 200MB. The engine indexes key sections dynamically."
        )
        if uploaded_file is not None:
            st.session_state.active_pdf = uploaded_file
            st.success(f"✅ Loaded: {uploaded_file.name} ({(uploaded_file.size)/(1024*1024):.1f} MB). You can now ask questions about this document below.")

# Bottom Chat Input
user_prompt = st.chat_input("Enter institutional query, model schedule, or analysis directive...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        status_box = st.status("Executing quantitative analysis...", expanded=False)

        def status_callback(status_type, message):
            status_box.write(f"• {message}")

        report, plot_bytes, excel_bytes, elapsed = agent.run_analyst_agent(
            user_input=user_prompt,
            api_key=groq_api_key,
            selected_model=selected_model,
            status_callback=status_callback,
            uploaded_pdf=st.session_state.active_pdf
        )

        status_box.update(label=f"✅ Complete ({elapsed:.2f}s)", state="complete")
        st.markdown(report)

        if plot_bytes:
            st.image(plot_bytes, caption="Valuation Artifact")
        if excel_bytes:
            st.download_button(
                label="📥 Download model.xlsx",
                data=excel_bytes,
                file_name="model.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": report,
        "plot_bytes": plot_bytes,
        "excel_bytes": excel_bytes,
        "timestamp": sys.float_info.max
    })
