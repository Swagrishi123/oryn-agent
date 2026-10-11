import streamlit as st
import math, re, urllib.parse, requests, io, sys, json
from pypdf import PdfReader
from groq import Groq
import sympy
from sympy import symbols, Symbol, Eq, solve, simplify, Rational

# --- Page Configuration ---
st.set_page_config(page_title="ORYN Agent", page_icon="✦", layout="centered")

# --- Custom UI Styling ---
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
        padding: 12px 0 8px 0;
        border-bottom: 1px solid rgba(0, 0, 0, 0.08);
        text-align: center;
        margin-bottom: 20px;
    }
    .oryn-brand {
        display: inline-flex;
        align-items: center;
        gap: 10px;
    }
    .oryn-title {
        font-size: 2rem;
        font-weight: 700;
        color: #4a148c !important;
        margin: 0;
    }
    .oryn-subtitle {
        color: #7b1fa2 !important;
        font-size: 0.85rem;
        margin: 2px 0 0 0;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background-color: #f3e8ff !important;
        border-radius: 18px !important;
        padding: 10px 16px !important;
        border: 1px solid #e9d5ff !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) * {
        color: #6b21a8 !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #faf5ff !important;
        border-radius: 18px !important;
        padding: 12px 18px !important;
        border: 1px solid #f3e8ff !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) *,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) p,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) span {
        color: #4a148c !important;
        font-weight: 500;
    }
    code {
        background-color: #ede9fe !important;
        color: #581c87 !important;
        font-weight: 600 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="oryn-header">
    <div class="oryn-brand">
        <svg width="30" height="30" viewBox="0 0 24 24" fill="#4a148c" xmlns="http://www.w3.org/2000/svg">
            <path d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.771-4.2057 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7467-7.0731zM13.2599 22.4042a4.4568 4.4568 0 0 1-2.9001-1.0543l.1422-.0818 4.811-2.778a.791.791 0 0 0 .3955-.6855v-6.786l2.1643 1.25a.1582.1582 0 0 1 .0791.137v5.545a4.4628 4.4628 0 0 1-4.692 4.4536zM3.8205 18.0645a4.4568 4.4568 0 0 1-.5157-3.0568l.1422.0853 4.811 2.778a.791.791 0 0 0 .791 0l5.8753-3.3934v2.5a.1582.1582 0 0 1-.0791.137l-4.8 2.7717a4.4628 4.4628 0 0 1-6.2247-1.8218zm-1.8218-9.4264a4.4568 4.4568 0 0 1 2.3844-2.0025v5.7176a.791.791 0 0 0 .3955.6855l5.8753 3.3934-2.1643 1.25a.1582.1582 0 0 1-.1582 0l-4.8-2.7717a4.4628 4.4628 0 0 1-1.5327-6.2723zm13.931-1.0543-5.8753-3.3934 2.1643-1.25a.1582.1582 0 0 1 .1582 0l4.8 2.7717a4.4628 4.4628 0 0 1-.5157 8.5292v-5.972a.791.791 0 0 0-.7315-.6855zm3.8205 6.3686a4.4568 4.4568 0 0 1 .5157 3.0568l-.1422-.0853-4.811-2.778a.791.791 0 0 0-.791 0l-5.8753 3.3934v-2.5a.1582.1582 0 0 1 .0791-.137l4.8-2.7717a4.4628 4.4628 0 0 1 6.2247 1.8218zM10.7401 13.0847l-2.6105-1.507 2.6105-1.507 2.6105 1.507z"/>
        </svg>
        <h1 class="oryn-title">ORYN AGENT</h1>
    </div>
    <p class="oryn-subtitle">Autonomous AI Agent • Multi-Tool Reasoning • Self-Correcting Execution</p>
</div>
""", unsafe_allow_html=True)

# --- API Key Setup & Dynamic Model Discovery ---
raw_key = st.secrets.get("GROQ_API_KEY", None)
if not raw_key:
    with st.sidebar:
        st.subheader("Configuration")
        raw_key = st.text_input("Enter Groq API Key:", type="password")
        st.caption("Free key available at console.groq.com")

api_key = str(raw_key).strip() if raw_key else None
client = Groq(api_key=api_key) if api_key else None

active_model = "llama-3.1-8b-instant"
if client:
    try:
        available_models = [m.id for m in client.models.list().data]
        priority_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-3.2-3b-preview",
            "mixtral-8x7b-32768"
        ]
        for candidate in priority_models:
            if candidate in available_models:
                active_model = candidate
                break
    except Exception as err:
        st.sidebar.warning(f"Could not load models list: {err}")

with st.sidebar:
    st.caption(f"⚡ Agent Core: `{active_model}`")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Tool Implementations ---
def tool_python_interpreter(code: str) -> str:
    """Executes Python and SymPy scripts safely, returning stdout."""
    stdout_trap = io.StringIO()
    sys_stdout_backup = sys.stdout
    scope = {
        "math": math, "sympy": sympy, "symbols": symbols, "Symbol": Symbol,
        "Eq": Eq, "solve": solve, "simplify": simplify, "Rational": Rational,
        "pi": math.pi, "__builtins__": __builtins__
    }
    try:
        sys.stdout = stdout_trap
        exec("import math\nimport sympy\nfrom sympy import symbols, Eq, solve, simplify, Rational\n" + code, scope)
        output = stdout_trap.getvalue().strip()
        return output if output else "Execution completed (no output printed)."
    except Exception as e:
        return f"Python Execution Error: {e}"
    finally:
        sys.stdout = sys_stdout_backup

def tool_wikipedia_search(query: str) -> str:
    """Searches Wikipedia for verified facts and summaries."""
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
        res = requests.get(url, headers={"User-Agent": "ORYN-Agent/2.0"}, timeout=4).json()
        hits = res.get("query", {}).get("search", [])
        if hits:
            clean_snippet = re.sub(r"<[^>]+>", "", hits[0].get("snippet", ""))
            return f"Wikipedia [{hits[0].get('title')}]: {clean_snippet}"
        return f"No Wikipedia results found for '{query}'."
    except Exception as e:
        return f"Wikipedia Search Error: {e}"

# --- Agent Tool Schema ---
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": "Executes Python / SymPy code to calculate exact arithmetic, algebra, percentages, or multi-step word problems deterministically. Outputs must use print().",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Executable Python code."}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Searches Wikipedia for verified real-world facts, current officeholders, statistics, and entities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."}
                },
                "required": ["query"]
            }
        }
    }
]

# --- Chat State & PDF Drawer ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.expander("📎 Attach Document (PDF)", expanded=False):
    uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"], label_visibility="collapsed")
    pdf_text = ""
    if uploaded_pdf:
        reader = PdfReader(uploaded_pdf)
        pdf_text = "".join([p.extract_text() or "" for p in reader.pages[:8]])[:4000]
        st.success(f"Attached: {uploaded_pdf.name}")

# Render Chat History
for msg in st.session_state.messages:
    if msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# --- User Input & ReAct Loop ---
user_input = st.chat_input("Ask ORYN Agent...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not client:
        with st.chat_message("assistant"):
            st.warning("Please provide a Groq API key in the sidebar to activate the agent.")
    else:
        system_instruction = (
            "You are ORYN Agent, an autonomous reasoning engine. "
            "Never guess math or real-world facts. Use `python_interpreter` for calculations and word problems. "
            "Use `wikipedia_search` for entities, roles, and statistics. "
            "If a tool call returns an error, analyze the trace and retry with corrected parameters. "
            "Synthesize a clear, direct explanation once tool executions conclude."
        )
        if pdf_text:
            system_instruction += f"\n\nContext from attached document:\n{pdf_text}"

        agent_messages = [{"role": "system", "content": system_instruction}]
        for m in st.session_state.messages[-4:]:
            if m.get("content"):
                agent_messages.append({"role": m["role"], "content": m["content"]})

        with st.chat_message("assistant"):
            status_container = st.status("🧠 ORYN Agent is analyzing intent...", expanded=True)
            max_iterations = 4
            iteration = 0
            final_content = ""

            while iteration < max_iterations:
                iteration += 1
                try:
                    response = client.chat.completions.create(
                        model=active_model,
                        messages=agent_messages,
                        tools=AGENT_TOOLS,
                        tool_choice="auto",
                        max_tokens=700
                    )
                except Exception as e:
                    final_content = f"API Error: {e}"
                    break

                response_message = response.choices[0].message
                agent_messages.append(response_message)

                if response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        func_name = tool_call.function.name
                        try:
                            args = json.loads(tool_call.function.arguments)
                        except Exception:
                            args = {}

                        status_container.write(f"🛠️ **Executing Tool:** `{func_name}`")

                        if func_name == "python_interpreter":
                            code_to_run = args.get("code", "")
                            status_container.code(code_to_run, language="python")
                            result = tool_python_interpreter(code_to_run)
                            status_container.write(f"**Output:** `{result}`")
                        elif func_name == "wikipedia_search":
                            q = args.get("query", "")
                            status_container.write(f"🔎 Query: *{q}*")
                            result = tool_wikipedia_search(q)
                            status_container.write(f"**Result:** {result}")
                        else:
                            result = f"Unknown tool '{func_name}'."

                        agent_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": result
                        })
                else:
                    final_content = response_message.content or getattr(response_message, "reasoning", "")
                    break

            status_container.update(label="✅ Reasoning complete", state="complete", expanded=False)
            
            if not final_content:
                final_content = "Process completed without output."

            st.markdown(final_content)
            st.session_state.messages.append({"role": "assistant", "content": final_content})
