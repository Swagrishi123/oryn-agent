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

# Benchmark Credit Agreement Template for 1-Click Verification
BENCHMARK_APEX_AGREEMENT = """CONFIDENTIAL CREDIT AGREEMENT & INDENTURE
BORROWER: APEX LOGISTICS HOLDINGS INC.
FACILITY: USD 450,000,000 SENIOR SECURED CREDIT FACILITIES
DATE: MARCH 15, 2026

SECTION 1.01: FINANCIAL DEFINITIONS & ADJUSTED EBITDA
"Consolidated Adjusted EBITDA" means, for any four-quarter testing period:
Consolidated Net Income plus:
(a) Consolidated Interest Expense ($42.5M for LTM);
(b) Provision for taxes based on income ($18.0M for LTM);
(c) Total Depreciation and Amortization ($35.5M for LTM);
(d) Non-recurring restructuring charges capped strictly at 10.0% of unadjusted EBITDA ($6.0M recognized).
Reported unadjusted LTM EBITDA is USD 80.0M.

SECTION 6.02: NEGATIVE COVENANTS - INDEBTEDNESS & LEVERAGE RATIOS
(a) Total Net Leverage Ratio: The Borrower shall not permit the Total Net Leverage Ratio as of the last day of any fiscal quarter to exceed 4.25x Consolidated Adjusted EBITDA.
(b) Senior Secured Net Leverage Ratio: The Borrower shall not permit the Senior Secured Net Leverage Ratio to exceed 3.00x Consolidated Adjusted EBITDA.
(c) Restricted Cash Exclusions: For purposes of calculating "Net Debt", unrestricted cash is capped at $50.0M for netting purposes. Restricted cash held in the collateral account ($18.5M) shall NOT be netted against Gross Debt under any circumstance.

SECTION 7.01: CAPITAL STRUCTURE AT CLOSING
- Term Loan A (Senior Secured): USD 180.0M (SOFR + 250 bps)
- Term Loan B (Senior Secured): USD 120.0M (SOFR + 350 bps)
- Senior Unsecured Notes: USD 100.0M (Fixed 8.25%)
- Revolving Credit Facility ($50M capacity): USD 15.0M drawn at closing
- Unrestricted Cash on Balance Sheet: USD 35.0M
- Restricted Collateral Cash: USD 18.5M"""

# Clean, safe styling with no viewport-covering z-index rules
st.markdown(
    """
    <style>
    /* Clean purple branding for the chat input */
    div[data-testid="stChatInput"] {
        border-radius: 20px !important;
        border: 1.5px solid #d8b4fe !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #7b1fa2 !important;
        box-shadow: 0 0 10px rgba(123, 31, 162, 0.2) !important;
    }
    .doc-drawer {
        background-color: #faf5ff;
        border: 1px solid #d8b4fe;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 10px;
    }
    .oryn-disclaimer {
        text-align: center;
        font-size: 11px;
        color: #8e8ea0;
        margin-top: 4px;
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
# SIDEBAR
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

    # Clean Session Controls: Wipes conversation to renew tokens, keeps active document
    st.subheader("🔄 Session Control")
    if st.button("🗑️ Reset Chat (Renew Tokens)", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_vault = []
        st.session_state.show_uploader = False
        st.rerun()

    if st.button("Purge Document & All Memory", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_vault = []
        st.session_state.active_pdf = None
        st.session_state.show_uploader = False
        st.rerun()

# ==========================================
# MAIN INTERACTION PANEL
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

# Render Chat History
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
# ATTACHMENT CONTROLS & DRAWER
# ==========================================
# Filing Indicator Badge
if st.session_state.active_pdf is not None:
    c_badge, c_detach = st.columns([6, 1])
    with c_badge:
        p_text = st.session_state.active_pdf.get("preview", "")
        c_count = st.session_state.active_pdf.get("chars", 0)
        if c_count < 100:
            st.warning(f"⚠️ **Loaded:** {st.session_state.active_pdf['name']} ({c_count} chars) — *File appears empty!*")
        else:
            st.info(f"📄 **Filing Loaded:** {st.session_state.active_pdf['name']} ({c_count:,} chars) • *Preview:* \"{p_text}...\"")
    with c_detach:
        if st.button("✕", help="Detach filing"):
            st.session_state.active_pdf = None
            st.rerun()

# Attachment Drawer Toggle
col_btn, _ = st.columns([1, 5])
with col_btn:
    btn_label = "✕ Close Attachment Panel" if st.session_state.show_uploader else "📎 Attach Filing / Document"
    if st.button(btn_label):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()

if st.session_state.show_uploader:
    with st.container():
        st.markdown(
            """
            <div class="doc-drawer">
                <span style="font-weight: 600; color: #4a148c; font-size: 13px;">
                    📑 Attach Institutional Filing (10-K, CIM, Credit Agreement up to 200MB)
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
        col_up, col_preset = st.columns([3, 2])
        with col_up:
            uploaded_file = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                label_visibility="collapsed",
                help="PDF filings are indexed dynamically."
            )
            if uploaded_file is not None:
                extracted_text, p_cnt, c_cnt = agent.extract_pdf_document_text(uploaded_file.getvalue())
                preview = extracted_text.replace("\n", " ").strip()[:90] if extracted_text else "No text found"
                st.session_state.active_pdf = {
                    "name": uploaded_file.name,
                    "size": uploaded_file.size,
                    "data": uploaded_file.getvalue(),
                    "preview": preview,
                    "chars": c_cnt
                }
                st.session_state.show_uploader = False
                st.rerun()

        with col_preset:
            if st.button("⚡ Load Apex Credit Agreement", help="Load the benchmark agreement directly into memory"):
                st.session_state.active_pdf = {
                    "name": "Apex_Credit_Agreement_Benchmark.txt",
                    "size": len(BENCHMARK_APEX_AGREEMENT.encode("utf-8")),
                    "text": BENCHMARK_APEX_AGREEMENT,
                    "preview": "CONFIDENTIAL CREDIT AGREEMENT & INDENTURE BORROWER: APEX LOGISTICS HOLDINGS INC...",
                    "chars": len(BENCHMARK_APEX_AGREEMENT)
                }
                st.session_state.show_uploader = False
                st.rerun()

# User Input
user_prompt = st.chat_input("Ask ORYN or run financial analysis...")
st.markdown('<div class="oryn-disclaimer">ORYN is an AI and can make mistakes. Verify critical financial data.</div>', unsafe_allow_html=True)

# ==========================================
# AGENT EXECUTION
# ==========================================
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        status_box = st.status("Executing quantitative analysis...", expanded=True)

        def status_callback(status_type, message):
            status_box.write(f"• {message}")

        report, plot_bytes, excel_bytes, elapsed = agent.run_analyst_agent(
            user_input=user_prompt,
            api_key=groq_api_key,
            selected_model=selected_model,
            status_callback=status_callback,
            uploaded_pdf=st.session_state.active_pdf
        )

        status_box.update(label=f"✅ Complete ({elapsed:.2f}s)", state="complete", expanded=False)
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
