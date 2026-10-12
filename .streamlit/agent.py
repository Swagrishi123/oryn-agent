import json
import re
import time
from groq import Groq
from tools import AGENT_TOOLS, tool_get_financial_data, tool_web_search
from sandbox import tool_python_interpreter

SYSTEM_PROMPT = (
    "You are ORYN Analyst, an institutional BFSI quantitative analyst built by Mriganka.\n"
    "CORE EXECUTION RULES:\n"
    "1. STRICT QUANTITATIVE FORMULAS:\n"
    "   - '% off 52W High' must ALWAYS be calculated as negative drawdown: ((Current Price - 52W High) / 52W High) * 100. Round all percentages to 2 decimals.\n"
    "   - For comps tables, sort rows logically (e.g., descending by Market Cap).\n"
    "2. PYTHON TOOL CALLING:\n"
    "   - When calling python_interpreter, use standard double quotes in print statements.\n"
    "   - Never escape single quotes with backslashes.\n"
    "3. NO LATEX & CLEAN TEXT FORMATTING:\n"
    "   - NEVER use LaTeX tags like \\approx, \\times, or math blocks ($$). Write plain text.\n"
    "   - NEVER output HTML tags like </br> or image placeholders.\n"
    "4. CURRENCY INTEGRITY:\n"
    "   - When comparing multi-currency companies, ALWAYS use 'Normalized Market Cap (USD)' for relative scale comparisons.\n"
    "   - For Indian domestic companies (.NS / .BO), state native metrics in Indian Rupees ('₹', Lakhs, Crores) alongside USD figures.\n"
    "5. REGULATORY & BANKING:\n"
    "   - For Indian banking (SBI, RBI, TDS, Section 194A, Form 15G/15H), ALWAYS use ₹ (INR), never $ (USD)."
)

def clean_extracted_code(raw_code: str) -> str:
    clean = raw_code.replace('\\\\n', '\n').replace('\\n', '\n')
    clean = clean.replace(r'\"', '"').replace(r"\'", "'").replace(r'\\', '\\')
    lines = clean.split('\n')
    valid_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped in ['PY', 'PY"', '}', '}}', '```', '```python']:
            continue
        if stripped.endswith('PY"') or stripped.endswith('PY'):
            line = line[:-3].rstrip('"')
        valid_lines.append(line)
    return '\n'.join(valid_lines).strip()

def extract_code_from_error(err_str: str) -> str | None:
    m_gen = re.search(r"['\"]failed_generation['\"]\s*:\s*['\"](.*?)['\"]\s*\}", err_str, re.DOTALL)
    search_target = m_gen.group(1) if m_gen else err_str

    m_code = re.search(r'["\']code["\']\s*:\s*["\'](.*?)(?:["\']\s*\}|\Z)', search_target, re.DOTALL)
    if m_code:
        return clean_extracted_code(m_code.group(1))
    return None

def run_analyst_agent(
    user_input: str,
    api_key: str,
    selected_model: str = "llama-3.1-8b-instant",
    hitl_enabled: bool = True,
    status_callback=None
) -> tuple[str, bytes | None, bytes | None, float]:
    start_time = time.perf_counter()
    client = Groq(api_key=api_key, max_retries=0)

    scratchpad = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    max_iterations = 6
    iteration = 0
    final_text = ""
    generated_plot = None
    generated_excel = None
    step_tool_count = 0
    active_model = "llama-3.1-8b-instant"

    while iteration < max_iterations:
        iteration += 1
        response = None
        last_error = ""

        try:
            response = client.chat.completions.create(
                model=active_model,
                messages=scratchpad,
                tools=AGENT_TOOLS,
                tool_choice="auto",
                max_tokens=2048
            )
        except Exception as e:
            last_error = str(e)
            
            # Auto-recovery: If Groq rejected single-quote escapes, extract and execute directly
            if "Failed to parse tool call arguments as JSON" in last_error or "tool_use_failed" in last_error:
                extracted_code = extract_code_from_error(last_error)
                if extracted_code:
                    if status_callback:
                        status_callback("info", "Executing quantitative calculations...")
                    out, p_bytes, x_bytes = tool_python_interpreter(extracted_code, hitl_enabled=hitl_enabled)
                    if p_bytes:
                        generated_plot = p_bytes
                    if x_bytes:
                        generated_excel = x_bytes
                    
                    scratchpad.append({
                        "role": "tool",
                        "tool_call_id": "call_recovered",
                        "name": "python_interpreter",
                        "content": out
                    })
                    step_tool_count += 1
                    break

        if not response and step_tool_count == 0:
            return f"Model execution halted: {last_error}", None, None, time.perf_counter() - start_time

        if response:
            msg = response.choices[0].message
            scratchpad.append(msg)

            if getattr(msg, "tool_calls", None):
                for tool_call in msg.tool_calls:
                    step_tool_count += 1
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except Exception:
                        fn_args = {}

                    if status_callback:
                        status_callback("info", f"Executing `{fn_name}`...")

                    if fn_name == "python_interpreter":
                        code_str = fn_args.get("code", "")
                        out, p_bytes, x_bytes = tool_python_interpreter(code_str, hitl_enabled=hitl_enabled)
                        if p_bytes:
                            generated_plot = p_bytes
                        if x_bytes:
                            generated_excel = x_bytes
                        tool_result = out

                    elif fn_name == "get_financial_data":
                        tk = fn_args.get("ticker", "")
                        tool_result = tool_get_financial_data(tk)

                    elif fn_name == "web_search":
                        tool_result = tool_web_search(fn_args.get("query", ""))

                    else:
                        tool_result = f"Unknown tool `{fn_name}`"

                    scratchpad.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": tool_result
                    })
            else:
                final_text = msg.content or ""
                break

    # Clean Synthesis Turn
    if step_tool_count > 0 and (not final_text or len(final_text.strip()) == 0):
        if status_callback:
            status_callback("info", "Synthesizing executive briefing...")

        tool_outputs = []
        for m in scratchpad:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role == "tool":
                name = m.get("name") if isinstance(m, dict) else getattr(m, "name", "Output")
                content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                tool_outputs.append(f"[{name}]:\n{content}")

        synth_messages = [
            {
                "role": "system",
                "content": (
                    "You are ORYN Analyst, an institutional BFSI quantitative analyst built by Mriganka. "
                    "Synthesize the user's inquiry and computed quantitative outputs into an executive valuation briefing. "
                    "Present clean markdown tables, valuation results, and institutional takeaways. "
                    "NEVER output raw code scripts, LaTeX blocks ($$), or HTML tags."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Inquiry: {user_input}\n\n"
                    f"Computed Data & Model Results:\n" + "\n\n".join(tool_outputs) + "\n\n"
                    "Deliver the final executive briefing in structured markdown now."
                )
            }
        ]

        try:
            synth_resp = client.chat.completions.create(
                model=active_model,
                messages=synth_messages,
                max_tokens=2048
            )
            final_text = synth_resp.choices[0].message.content or "Analysis completed."
        except Exception as e:
            final_text = f"Synthesis note: {e}"

    elapsed = time.perf_counter() - start_time
    return final_text, generated_plot, generated_excel, elapsed
