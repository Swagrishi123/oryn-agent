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

# Custom CSS: Docks the '+' button directly inside the left edge of the chat input box
st.markdown(
    """
    <style>
    /* Ensure the bottom container allows absolute positioning */
    div[data-testid="stBottom"] > div {
        position: relative !important;
    }

    /* Indent chat input text so it starts after the '+' button */
    div[data-testid="stChatInput"] textarea {
        padding-left: 52px !important;
    }

    /* Position the '+' button directly inside the left of the input bar */
    .gemini-attach-wrapper {
        position: absolute !important;
        bottom: 21px !important;
        left: 10px !important;
        z-index: 99999 !important;
    }

    .gemini-attach-wrapper div[data-testid="stButton"] {
        margin: 0 !important;
        padding: 0 !important;
    }

    .gemini-attach-wrapper div[data-testid="stButton"] > button {
        border-radius: 50% !important;
        width: 34px !important;
        height: 34px !important;
        min-height: 34px !important;
        padding: 0 !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        line-height: 1 !important;
        border: 1px solid #7b1fa2 !important;
        background-color: #faf5ff !important;
        color: #4a148c !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.2s !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }

    .gemini-attach-wrapper div[data-testid="stButton"] > button:hover {
        background-color: #f3e8ff !important;
        border-color: #4a148c !important;
    }

    /* Rotate into an 'X' when the drawer is open */
    .gemini-attach-wrapper.is-open div[data-testid="stButton"] > button {
        transform: rotate(45deg) !important;
        background-color: #4a148c !important;
        color: #ffffff !important;
        border-color: #4a148c !important;
    }

    /* Document Drawer Styling */
    .doc-drawer {
        background-color: #ffffff;
        border: 1px solid #e9d5ff;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
        box-shadow: 0 4px 12px rgba(74, 20, 140, 0.08);
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
# GEMINI-STYLE BOTTOM DOCK (INPUT & DRAWER)
# ==========================================
# Use st.bottom directly without calling ()
bottom_area = st.bottom if hasattr(st, "bottom") else st.container()

with bottom_area:
    # 1. Expandable PDF Drawer (Opens above chat bar when '+' is clicked)
    if st.session_state.show_uploader:
        with st.container():
            st.markdown(
                """
                <div class="doc-drawer">
                    <span style="font-weight: 600; color: #4a148c; font-size: 14px;">
                        📑 Attach Institutional Filing (10-K, Credit Indenture, CIM up to 200MB)
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            uploaded_file = st.file_uploader(
                "Upload 200MB PDF",
                type=["pdf"],
                label_visibility="collapsed",
                help="Dense filings up to 200MB are indexed dynamically."
            )
            if uploaded_file is not None:
                st.session_state.active_pdf = uploaded_file
                st.success(f"✅ Attached: {uploaded_file.name} ({(uploaded_file.size)/(1024*1024):.1f} MB)")

    # 2. Attached Document Indicator Chip
    if st.session_state.active_pdf is not None and not st.session_state.show_uploader:
        chip_col1, chip_col2 = st.columns([5, 1])
        with chip_col1:
            st.info(f"📄 **Filing Loaded:** {st.session_state.active_pdf.name} ({(st.session_state.active_pdf.size)/(1024*1024):.1f} MB)")
        with chip_col2:
            if st.button("Detach ✕", help="Remove attached document"):
                st.session_state.active_pdf = None
                st.rerun()

    # 3. Rotating '+' Button (Anchored directly inside the left of the input bar)
    btn_state_class = "is-open" if st.session_state.show_uploader else ""
    st.markdown(f'<div class="gemini-attach-wrapper {btn_state_class}">', unsafe_allow_html=True)
    if st.button("➕", key="gemini_plus_btn", help="Attach 10-K, CIM, or Credit Agreement"):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. Bottom Chat Input
    user_prompt = st.chat_input("Enter institutional query, model schedule, or analysis directive...")

# ==========================================
# AGENT EXECUTION
# ==========================================
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
