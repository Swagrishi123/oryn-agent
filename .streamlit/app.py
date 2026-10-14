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

# Custom CSS: Docks the '+' button directly INSIDE the typing bar (Gemini-style)
st.markdown(
    """
    <style>
    /* Make the bottom container allow precise positioning */
    div[data-testid="stBottom"] > div,
    div[data-testid="stBottomBlockContainer"] {
        position: relative !important;
    }

    /* Indent chat placeholder and text so it starts after the '+' button */
    div[data-testid="stChatInput"] textarea {
        padding-left: 54px !important;
    }

    /* Pull the attach button directly inside the left edge of the typing pill */
    div[data-testid="stBottom"] div[data-testid="stElementContainer"]:has(#gemini_attach_anchor),
    div[data-testid="stBottom"] div[data-testid="stElementContainer"]:has(div[data-testid="stButton"]) {
        position: absolute !important;
        bottom: 22px !important;
        left: 14px !important;
        z-index: 99999 !important;
        width: 32px !important;
        height: 32px !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Style the '+' button to match Gemini: borderless, circular, centered */
    div[data-testid="stBottom"] div[data-testid="stButton"] > button {
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        min-height: 32px !important;
        max-height: 32px !important;
        border: none !important;
        background-color: transparent !important;
        color: #7b1fa2 !important;
        font-size: 20px !important;
        font-weight: 500 !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        margin: 0 !important;
        transition: background-color 0.2s ease, transform 0.2s ease !important;
    }

    div[data-testid="stBottom"] div[data-testid="stButton"] > button:hover {
        background-color: rgba(123, 31, 162, 0.12) !important;
        color: #4a148c !important;
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
bottom_area = st.bottom if hasattr(st, "bottom") else st.container()

with bottom_area:
    # 1. Expandable PDF Drawer (Slides open above the typing bar)
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
        st.caption(f"📄 **Filing Loaded:** {st.session_state.active_pdf.name} ({(st.session_state.active_pdf.size)/(1024*1024):.1f} MB) — *Click ➕ to change document*")

    # 3. Rotating Gemini '+' Attachment Button (Anchored inside the typing bar)
    btn_icon = "✕" if st.session_state.show_uploader else "➕"
    if st.button(btn_icon, key="gemini_attach_btn", help="Attach 10-K, CIM, or Credit Agreement"):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()

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
