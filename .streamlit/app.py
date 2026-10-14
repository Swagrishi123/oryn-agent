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

# Custom CSS: Ultra-sleek, compact typing pill calibrated for mobile and desktop
st.markdown(
    """
    <style>
    /* Container setup for absolute button docking */
    div[data-testid="stBottom"] > div,
    div[data-testid="stBottomBlockContainer"] {
        position: relative !important;
        background-color: transparent !important;
        padding-bottom: 6px !important;
    }

    /* Force the typing pill to be thin, sleek, and compact (44px height) */
    div[data-testid="stChatInput"] {
        border-radius: 26px !important;
        background-color: #ffffff !important;
        border: 1.5px solid #d8b4fe !important;
        box-shadow: 0 2px 10px rgba(74, 20, 140, 0.07) !important;
        min-height: 44px !important;
        max-height: 44px !important;
        height: 44px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
        transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: #7b1fa2 !important;
        box-shadow: 0 4px 16px rgba(74, 20, 140, 0.14) !important;
    }

    /* Single-line compact textarea with left indent for the '+' button */
    div[data-testid="stChatInput"] textarea {
        padding-left: 48px !important;
        padding-right: 42px !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        height: 42px !important;
        font-size: 14px !important;
        line-height: 22px !important;
        resize: none !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }

    /* Floating Send Button centered inside the thin pill */
    div[data-testid="stChatInput"] button {
        border-radius: 50% !important;
        width: 30px !important;
        height: 30px !important;
        min-height: 30px !important;
        max-height: 30px !important;
        background-color: #faf5ff !important;
        color: #4a148c !important;
        border: 1px solid #d8b4fe !important;
        box-shadow: 0 1px 4px rgba(74, 20, 140, 0.15) !important;
        margin-top: auto !important;
        margin-bottom: auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* Precise anchor: Docks the '+' button directly inside the left of the 44px pill */
    .st-key-gemini_attach_btn {
        position: absolute !important;
        left: 12px !important;
        bottom: 34px !important;
        z-index: 99999 !important;
        width: 30px !important;
        height: 30px !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    .st-key-gemini_attach_btn > div,
    .st-key-gemini_attach_btn button {
        border-radius: 50% !important;
        width: 30px !important;
        height: 30px !important;
        min-height: 30px !important;
        max-height: 30px !important;
        border: 1px solid #d8b4fe !important;
        background-color: #faf5ff !important;
        color: #4a148c !important;
        font-size: 17px !important;
        font-weight: 600 !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: 0 1px 4px rgba(74, 20, 140, 0.12) !important;
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.2s !important;
    }

    .st-key-gemini_attach_btn button:hover {
        background-color: #ede9fe !important;
        border-color: #7b1fa2 !important;
        color: #3b0764 !important;
    }

    /* Document Drawer Styling */
    .doc-drawer {
        background-color: #ffffff;
        border: 1px solid #e9d5ff;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 8px;
        box-shadow: 0 4px 16px rgba(74, 20, 140, 0.08);
    }

    /* Discreet Disclaimer */
    .oryn-disclaimer {
        text-align: center;
        margin-top: 5px;
        margin-bottom: 2px;
        font-size: 11px;
        color: #8e8ea0;
        letter-spacing: 0.1px;
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
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
            <span style="font-size: 28px;">🏛️</span>
            <span style="font-size: 22px; font-weight: 700; color: #4a148c;">ORYN Analyst</span>
        </div>
        """,
        unsafe_allow_html=True
    )
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
# MAIN INTERACTION PANEL (WITH ORYN EMBLEM)
# ==========================================
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 1.2rem;">
        <span style="font-size: 34px;">🏛️</span>
        <div>
            <h2 style="margin: 0; color: #4a148c; font-weight: 700; font-size: 26px; line-height: 1.2;">
                ORYN Analyst
            </h2>
            <p style="margin: 0; color: #6b7280; font-size: 14px;">
                Institutional BFSI Quantitative Intelligence & Workspace
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

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
# GEMINI-STYLE BOTTOM DOCK (COMPACT UI)
# ==========================================
bottom_area = st.bottom if hasattr(st, "bottom") else st.container()

with bottom_area:
    # 1. Expandable PDF Drawer (Slides open above the typing bar)
    if st.session_state.show_uploader:
        with st.container():
            st.markdown(
                """
                <div class="doc-drawer">
                    <span style="font-weight: 600; color: #4a148c; font-size: 13.5px;">
                        📑 Attach Institutional Filing (up to 200MB)
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
        st.caption(f"📄 **Filing Loaded:** {st.session_state.active_pdf.name} ({(st.session_state.active_pdf.size)/(1024*1024):.1f} MB) — *Click ➕ to change*")

    # 3. Gemini '+' Attachment Button (Anchored by key into left edge)
    btn_icon = "✕" if st.session_state.show_uploader else "➕"
    if st.button(btn_icon, key="gemini_attach_btn", help="Attach 10-K, CIM, or Credit Agreement"):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()

    # 4. Single-line compact placeholder (prevents multi-line expansion on mobile)
    user_prompt = st.chat_input("Ask ORYN or attach filing...")

    # 5. Clean AI Disclaimer
    st.markdown(
        """
        <div class="oryn-disclaimer">
            ORYN is an AI and can make mistakes. Verify critical financial data.
        </div>
        """,
        unsafe_allow_html=True
    )

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
