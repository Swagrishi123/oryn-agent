import json
import os
import re
import time
from groq import Groq
from tools import AGENT_TOOLS, tool_get_financial_data, tool_web_search
from sandbox import tool_python_interpreter

SYSTEM_PROMPT = (
    "You are ORYN Analyst, an institutional BFSI quantitative analyst.\n"
    "CORE RULES:\n"
    "1. For valuation and DCF models, compute deterministic cash flows, discounting, and sensitivity matrices using python_interpreter.\n"
    "2. Round all percentages to 2 decimal places and currency values appropriately.\n"
    "3. Output clean Markdown tables and executive summaries. Do not include raw code scripts, LaTeX blocks ($$), or HTML tags in the final briefing."
)

def clean_extracted_code(raw_code: str) -> str:
    clean = raw_code.replace('\\\\n', '\n').replace('\\n', '\n')
    clean = clean.replace(r'\"', '"').replace(r"\'", "'").replace(r'\\', '\\')
    lines = clean.split('\n')
    valid = [l for l in lines if l.strip() not in ['PY', 'PY"', '}', '}}', '```', '```python']]
    return '\n'.join(valid).strip()

def extract_code_from_error(err_str: str) -> str | None:
    m_gen = re.search(r"['\"]failed_generation['\"]\s*:\s*['\"](.*?)['\"]\s*\}", err_str, re.DOTALL)
    search_target = m_gen.group(1) if m_gen else err_str
    m_code = re.search(r'["\']code["\']\s*:\s*["\'](.*?)(?:["\']\s*\}|\Z)', search_target, re.DOTALL)
    return clean_extracted_code(m_code.group(1)) if m_code else None

def resolve_api_key(api_key: str = "") -> str:
    # 1. User sidebar input
    if api_key and api_key.strip():
        return api_key.strip()
    # 2. Streamlit Cloud Secrets
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    # 3. Environment Variable
    return os.environ.get("GROQ_API_KEY", "").strip()

def run_analyst_agent(
    user_input: str,
    api_key: str = "",
    engine_mode: str = "Cloud (Groq BYOK)",
    hitl_enabled: bool = True,
    status_callback=None,
    **kwargs
) -> tuple[str, bytes | None, bytes | None, float]:
    start_time = time.perf_counter()

    active_key = resolve_api_key(api_key)
    if not active_key:
        return (
            "⚠️ **API Key Missing**: No Groq API key found. Please enter your free key in the sidebar "
            "or configure `GROQ_API_KEY` under Streamlit Cloud App Settings → Secrets.",
            None,
            None,
            0.0
        )

    try:
        client = Groq(api_key=active_key, max_retries=0)
    except Exception as e:
        return f"⚠️ **Client Initialization Error**: {e}", None, None, 0.0

    active_model = "llama-3.3-70b-versatile"

    scratchpad = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    generated_plot = None
    generated_excel = None
    tool_outputs = []
    final_text = ""

    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=active_model,
                messages=scratchpad,
                tools=AGENT_TOOLS,
                tool_choice="auto",
                max_tokens=2048
            )
        except Exception as e:
            err = str(e)
            recovered = extract_code_from_error(err)
            if recovered:
                if status_callback:
                    status_callback("info", "Executing recovered quantitative code...")
                out, p_bytes, x_bytes = tool_python_interpreter(recovered, hitl_enabled=hitl_enabled)
                if p_bytes:
                    generated_plot = p_bytes
                if x_bytes:
                    generated_excel = x_bytes
                tool_outputs.append(f"[python_interpreter]:\n{out}")
                scratchpad.append({"role": "tool", "tool_call_id": "call_rec", "name": "python_interpreter", "content": out})
                break
            return f"Execution error: {err}", None, None, time.perf_counter() - start_time

        msg = response.choices[0].message
        scratchpad.append(msg)

        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}

                if status_callback:
                    status_callback("info", f"Executing `{fn_name}`...")

                if fn_name == "python_interpreter":
                    out, p_bytes, x_bytes = tool_python_interpreter(args.get("code", ""), hitl_enabled=hitl_enabled)
                    if p_bytes:
                        generated_plot = p_bytes
                    if x_bytes:
                        generated_excel = x_bytes
                    res = out
                elif fn_name == "get_financial_data":
                    res = tool_get_financial_data(args.get("ticker", ""))
                elif fn_name == "web_search":
                    res = tool_web_search(args.get("query", ""))
                else:
                    res = f"Unknown tool: {fn_name}"

                tool_outputs.append(f"[{fn_name}]:\n{res}")
                scratchpad.append({"role": "tool", "tool_call_id": tc.id, "name": fn_name, "content": res})
        else:
            final_text = msg.content or ""
            break

    # Final Executive Synthesis
    if tool_outputs and not final_text:
        if status_callback:
            status_callback("info", "Generating executive valuation briefing...")
        try:
            synth = client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Query: {user_input}\n\nTool Results:\n" + "\n\n".join(tool_outputs) + "\n\nProvide the final executive valuation briefing in clean Markdown tables."}
                ],
                max_tokens=2048
            )
            final_text = synth.choices[0].message.content or "Analysis complete."
        except Exception as e:
            final_text = f"Synthesis note: {e}"

    elapsed = time.perf_counter() - start_time
    return final_text, generated_plot, generated_excel, elapsed
