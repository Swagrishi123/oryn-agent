import streamlit as st
import math, re, urllib.parse, requests, io, sys, json, time, base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pypdf import PdfReader
from groq import Groq
import sympy
from sympy import symbols, Symbol, Eq, solve, simplify, Rational
from duckduckgo_search import DDGS

# --- Page Configuration (Mobile-Optimized) ---
st.set_page_config(
    page_title="ORYN Agent — Unified",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Responsive Purple UI & Mobile Styling ---
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
        <h1 class="oryn-title">ORYN AGENT</h1>
    </div>
    <p class="oryn-subtitle">Autonomous Reasoning • Tool Chaining • HITL Governance</p>
</div>
""", unsafe_allow_html=True)

# --- State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = st.secrets.get("GROQ_API_KEY", "")

if "telemetry" not in st.session_state:
    st.session_state.telemetry = {
        "queries": 0,
        "tool_calls": 0,
        "last_latency_s": 0.0,
        "tokens_est": 0,
        "breakdown": {"python": 0, "web": 0, "wiki": 0}
    }

# Top reliable models prioritizing tool-use capabilities
PREFERRED_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b"
]

# Quick Reset Bar
col_hdr_l, col_hdr_r = st.columns([3, 1])
with col_hdr_r:
    if st.button("🗑️ Reset", use_container_width=True, help="Clear chat"):
        st.session_state.messages = []
        st.session_state.telemetry = {
            "queries": 0, "tool_calls": 0, "last_latency_s": 0.0, "tokens_est": 0,
            "breakdown": {"python": 0, "web": 0, "wiki": 0}
        }
        st.rerun()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Agent Controls")
    
    key_input = st.text_input(
        "Groq API Key:",
        value=st.session_state.api_key,
        type="password",
        help="Permanent key: Streamlit Cloud -> Settings -> Secrets"
    )
    if key_input:
        st.session_state.api_key = key_input.strip()

    client = Groq(api_key=st.session_state.api_key) if st.session_state.api_key else None

    # Discover account models and sort tool-friendly ones to the top
    available_models = []
    if client:
        try:
            raw_models = [m.id for m in client.models.list().data]
            banned = ["whisper", "guard", "vision", "embed", "safeguard", "orpheus", "canopy", "audio", "tts"]
            available_models = [m for m in raw_models if not any(b in m.lower() for b in banned)]
            
            def sort_models(m):
                m_low = m.lower()
                if "3.1-8b" in m_low: return 0
                if "3.3-70b" in m_low: return 1
                if "70b" in m_low: return 2
                return 10
            available_models.sort(key=sort_models)
        except Exception:
            available_models = []

    if not available_models:
        available_models = PREFERRED_MODELS

    active_model = st.selectbox("Active Model Engine:", options=available_models, index=0)
    hitl_enabled = st.toggle("🛡️ HITL Safety Guardrails", value=True)

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
    st.text(f"Python: {b['python']} | Web: {b['web']} | Wiki: {b['wiki']}")

# --- Sanitization Helper ---
def clean_code_snippet(code: str) -> str:
    cleaned_lines = []
    for line in code.strip().splitlines():
        trimmed = line.strip()
        if trimmed.startswith("```"):
            continue
        if "plt.show()" in trimmed:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

# --- Tool Implementations ---
def tool_python_interpreter(code: str) -> tuple[str, bytes | None]:
    code = clean_code_snippet(code)
    
    if hitl_enabled:
        forbidden = ["import os", "import sys", "import subprocess", "import shutil", "__import__", "open("]
        for term in forbidden:
            if term in code:
                return f"🛡️ HITL Intercept: Execution blocked. '{term}' is restricted.", None

    stdout_trap = io.StringIO()
    sys_stdout_backup = sys.stdout
    plot_bytes = None
    
    scope = {
        "math": math, "sympy": sympy, "symbols": symbols, "Symbol": Symbol,
        "Eq": Eq, "solve": solve, "simplify": simplify, "Rational": Rational,
        "pi": math.pi, "plt": plt, "np": np, "__builtins__": __builtins__
    }

    try:
        plt.close('all')
        sys.stdout = stdout_trap
        exec("import math\nimport sympy\nimport numpy as np\nfrom sympy import symbols, Eq, solve, simplify, Rational\nimport matplotlib.pyplot as plt\n" + code, scope)
        
        if plt.get_fignums():
            fig = plt.gcf()
            fig_buf = io.BytesIO()
            fig.savefig(fig_buf, format="png", bbox_inches="tight", dpi=140)
            fig_buf.seek(0)
            plot_bytes = fig_buf.getvalue()
            plt.close('all')

        output = stdout_trap.getvalue().strip()
        return (output if output else "Plot generated successfully."), plot_bytes
    except Exception as e:
        plt.close('all')
        return f"Python Execution Error: {e}", None
    finally:
        sys.stdout = sys_stdout_backup

def tool_web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"No web search results found for '{query}'."
        formatted = []
        for r in results:
            formatted.append(f"• [{r.get('title', '')}]: {r.get('body', '')[:280]}")
        return "\n".join(formatted)
    except Exception as e:
        return f"Web Search Error: {e}"

def tool_wikipedia_search(query: str) -> str:
    try:
        url = f"[https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=](https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=){urllib.parse.quote(query)}&utf8=&format=json"
        res = requests.get(url, headers={"User-Agent": "ORYN-Agent/3.0"}, timeout=4).json()
        hits = res.get("query", {}).get("search", [])
        if hits:
            clean = re.sub(r"<[^>]+>", "", hits[0].get("snippet", ""))
            return f"Wikipedia [{hits[0].get('title')}]: {clean}"
        return f"No Wikipedia entries found for '{query}'."
    except Exception as e:
        return f"Wikipedia Error: {e}"

def build_html_report(text: str, plot_bytes: bytes | None) -> str:
    img_tag = ""
    if plot_bytes:
        b64 = base64.b64encode(plot_bytes).decode('utf-8')
        img_tag = f'<div style="text-align:center; margin:20px 0;"><img src="data:image/png;base64,{b64}" style="max-width:100%; border-radius:12px; border:1px solid #ddd;"/></div>'
    
    clean_body = text.replace('\n', '<br>')
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ORYN Agent Executive Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 15px auto; max-width: 800px; padding: 15px; color: #2d0b5a; background-color: #faf5ff; }}
        .card {{ background: white; padding: 20px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); border: 1px solid #e9d5ff; }}
        h1 {{ color: #4a148c; border-bottom: 2px solid #e9d5ff; padding-bottom: 10px; font-size: 1.4rem; }}
        .content {{ font-size: 14px; line-height: 1.6; color: #333; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>ORYN Agent — Intelligence Report</h1>
        {img_tag}
        <div class="content">{clean_body}</div>
    </div>
</body>
</html>"""

# --- Tool Schemas ---
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": "Executes Python code for exact calculations or charts with Matplotlib (plt.plot, plt.bar). Provide executable Python code only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Executable Python code string."}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the live web for recent updates, real-time facts, and market information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Searches Wikipedia for verified background facts, history, and encyclopedic definitions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term."}
                },
                "required": ["query"]
            }
        }
    }
]

# --- Ingest Drawer ---
with st.expander("📎 Ingest Document (PDF)", expanded=False):
    uploaded_pdf = st.file_uploader("Attach PDF", type=["pdf"], label_visibility="collapsed")
    pdf_text = ""
    if uploaded_pdf:
        reader = PdfReader(uploaded_pdf)
        pdf_text = "".join([p.extract_text() or "" for p in reader.pages[:8]])[:4000]
        st.success(f"Ingested: {uploaded_pdf.name}")

# --- Render Chat History ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"], caption="Rendered Visual", use_container_width=True)
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    label="📊 Download Chart (.png)",
                    data=msg["image"],
                    file_name=f"oryn_chart_{idx}.png",
                    mime="image/png",
                    key=f"dl_png_hist_{idx}",
                    use_container_width=True
                )
            with col_d2:
                html_doc = build_html_report(msg["content"], msg["image"])
                st.download_button(
                    label="📑 Download Report (.html)",
                    data=html_doc,
                    file_name=f"oryn_report_{idx}.html",
                    mime="text/html",
                    key=f"dl_html_hist_{idx}",
                    use_container_width=True
                )
        elif msg.get("download"):
            st.download_button(
                label="📥 Download Output (.txt)",
                data=msg["content"],
                file_name=f"oryn_report_{idx}.txt",
                mime="text/plain",
                key=f"dl_text_hist_{idx}",
                use_container_width=True
            )

# --- User Input & Resilient ReAct Engine ---
user_input = st.chat_input("Ask ORYN Agent (e.g. plot data, calculate math, search web)...")

if user_input:
    start_time = time.perf_counter()
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not client:
        with st.chat_message("assistant"):
            st.error("Please provide a Groq API Key in the sidebar or via Streamlit Secrets.")
    else:
        system_prompt = (
            "You are ORYN Agent, an independent autonomous AI system built by Mriganka.\n"
            "IDENTITY RULES:\n"
            "- Never identify as ChatGPT or state that you were created by OpenAI.\n"
            "- If asked about privacy or your owner, clarify that you are ORYN Agent running an independent session without third-party monitoring.\n"
            "TOOL RULES:\n"
            "- If the user asks for a chart, graph, plot, or calculation, call `python_interpreter` or output executable Python code.\n"
            "- Deliver clear, direct answers and concise executive summaries."
        )
        if pdf_text:
            system_prompt += f"\n\nContext from ingested PDF:\n{pdf_text}"

        agent_messages = [{"role": "system", "content": system_prompt}]
        for m in st.session_state.messages[-4:]:
            if m.get("content"):
                agent_messages.append({"role": m["role"], "content": m["content"]})

        with st.chat_message("assistant"):
            status_container = st.status("🧠 ORYN Agent is planning & executing...", expanded=True)
            max_iterations = 4
            iteration = 0
            final_text = ""
            generated_plot = None
            step_tool_count = 0

            while iteration < max_iterations:
                iteration += 1
                response = None
                
                # Attempt 1: Call with Tool Definitions
                try:
                    response = client.chat.completions.create(
                        model=active_model,
                        messages=agent_messages,
                        tools=AGENT_TOOLS,
                        tool_choice="auto",
                        max_tokens=850
                    )
                except Exception as err:
                    err_msg = str(err).lower()
                    # If model rejects tool calling (HTTP 400), silently fallback to direct conversation without tools
                    if "tool" in err_msg or "400" in err_msg or "param" in err_msg:
                        try:
                            response = client.chat.completions.create(
                                model=active_model,
                                messages=agent_messages,
                                max_tokens=850
                            )
                        except Exception as inner_err:
                            final_text = f"Groq Engine Error: {inner_err}"
                            break
                    else:
                        final_text = f"Groq Engine Error: {err}"
                        break

                if not response:
                    break

                response_message = response.choices[0].message
                agent_messages.append(response_message)

                # Process Tool Invocations if returned
                if getattr(response_message, "tool_calls", None):
                    for tool_call in response_message.tool_calls:
                        step_tool_count += 1
                        fn_name = tool_call.function.name
                        try:
                            fn_args = json.loads(tool_call.function.arguments)
                        except Exception:
                            fn_args = {}

                        status_container.write(f"🛠️ **Executing Tool:** `{fn_name}`")

                        if fn_name == "python_interpreter":
                            code_str = fn_args.get("code", "")
                            status_container.code(clean_code_snippet(code_str), language="python")
                            out, maybe_plot = tool_python_interpreter(code_str)
                            if maybe_plot:
                                generated_plot = maybe_plot
                                status_container.write("📊 *Plot generated successfully.*")
                            status_container.write(f"**Output:** `{out}`")
                            tool_result = out
                            st.session_state.telemetry["breakdown"]["python"] += 1

                        elif fn_name == "web_search":
                            q = fn_args.get("query", "")
                            status_container.write(f"🌐 Query: *{q}*")
                            tool_result = tool_web_search(q)
                            status_container.write(tool_result)
                            st.session_state.telemetry["breakdown"]["web"] += 1

                        elif fn_name == "wikipedia_search":
                            q = fn_args.get("query", "")
                            status_container.write(f"📖 Query: *{q}*")
                            tool_result = tool_wikipedia_search(q)
                            status_container.write(tool_result)
                            st.session_state.telemetry["breakdown"]["wiki"] += 1

                        else:
                            tool_result = f"Unknown tool `{fn_name}`."

                        agent_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": fn_name,
                            "content": tool_result
                        })
                else:
                    final_text = response_message.content or getattr(response_message, "reasoning", "")
                    break

            # Fallback Interceptor: Execute embedded Python code if model wrote markdown
            if not generated_plot and ("```python" in final_text or "plt." in final_text):
                status_container.write("⚡ *Executing inline plot script...*")
                code_matches = re.findall(r"```(?:python)?\s*(.*?)\s*```", final_text, re.DOTALL)
                target_code = "\n".join(code_matches) if code_matches else final_text
                out, maybe_plot = tool_python_interpreter(target_code)
                if maybe_plot:
                    generated_plot = maybe_plot
                    status_container.write("📊 *Plot generated via fallback.*")

            elapsed = time.perf_counter() - start_time
            status_container.update(label=f"✅ Execution Complete ({elapsed:.2f}s)", state="complete", expanded=False)

            if not final_text:
                final_text = "Task completed."

            # Update Telemetry
            st.session_state.telemetry["queries"] += 1
            st.session_state.telemetry["tool_calls"] += step_tool_count
            st.session_state.telemetry["last_latency_s"] = elapsed
            st.session_state.telemetry["tokens_est"] += len(user_input.split()) * 2 + len(final_text.split()) * 2

            # Render Responses
            st.markdown(final_text)
            if generated_plot:
                st.image(generated_plot, caption="Rendered Visual", use_container_width=True)
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.download_button(
                        label="📊 Download Chart (.png)",
                        data=generated_plot,
                        file_name="oryn_chart.png",
                        mime="image/png",
                        key=f"dl_live_png_{len(st.session_state.messages)}",
                        use_container_width=True
                    )
                with col_d2:
                    html_report = build_html_report(final_text, generated_plot)
                    st.download_button(
                        label="📑 Download Report (.html)",
                        data=html_report,
                        file_name="oryn_executive_report.html",
                        mime="text/html",
                        key=f"dl_live_html_{len(st.session_state.messages)}",
                        use_container_width=True
                    )
            else:
                st.download_button(
                    label="📥 Download Output (.txt)",
                    data=final_text,
                    file_name="oryn_report.txt",
                    mime="text/plain",
                    key=f"dl_live_txt_{len(st.session_state.messages)}",
                    use_container_width=True
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_text,
                "image": generated_plot,
                "download": True
            })
