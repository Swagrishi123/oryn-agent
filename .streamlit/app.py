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

if "working_memory" not in st.session_state:
    st.session_state.working_memory = []

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []

if "image_insights" not in st.session_state:
    st.session_state.image_insights = ""

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
        st.session_state.telemetry = {
            "queries": 0, "tool_calls": 0, "last_latency_s": 0.0, "tokens_est": 0,
            "breakdown": {"python": 0, "web": 0, "wiki": 0, "finance": 0, "doc_search": 0}
        }
        st.rerun()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Analyst Controls")
    key_input = st.text_input(
        "Groq API Key:",
        value=st.session_state.api_key,
        type="password",
        help="Permanent key: Streamlit Cloud -> Settings -> Secrets"
    )
    if key_input:
        st.session_state.api_key = key_input.strip()

    client = Groq(api_key=st.session_state.api_key) if st.session_state.api_key else None

    # Discover live active models dynamically
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
