import json
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
    "2. NO LATEX & CLEAN TEXT FORMATTING:\n"
    "   - NEVER use LaTeX tags like \\approx, \\times, or math blocks ($$). Write plain text (e.g., 'approx. 15%').\n"
    "   - NEVER output HTML tags like </br> or image placeholders like '[Image: ...]'.\n"
    "   - NEVER dump Python scripts into text; run code silently using python_interpreter.\n"
    "3. CURRENCY INTEGRITY:\n"
    "   - When comparing multi-currency companies, ALWAYS use 'Normalized Market Cap (USD)' for relative scale comparisons.\n"
    "   - For Indian domestic companies (.NS / .BO), state native metrics in Indian Rupees ('₹', Lakhs, Crores) alongside USD figures.\n"
    "4. REGULATORY & BANKING:\n"
    "   - For Indian banking (SBI, RBI, TDS, Section 194A, Form 15G/15H), ALWAYS use ₹ (INR), never $ (USD)."
)

def run_analyst_agent(
    user_input: str,
    api_key: str,
    selected_model: str = "llama-3.1-8b-instant",
    hitl_enabled: bool = True,
    status_callback=None
) -> tuple[str, bytes | None, bytes | None, float]:
    """
    Executes the autonomous loop in an isolated, token-efficient scratchpad.
    Returns: (final_text, plot_bytes, excel_bytes, elapsed_seconds)
    """
    start_time = time.perf_counter()
    client = Groq(api_key=api_key, max_retries=0)

    # 1. Ephemeral scratchpad: never accumulates multi-turn conversation bloat
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

    # Model fallback priority to dodge rate limits
    model_queue = [selected_model]
    if "llama-3.1-8b-instant" not in model_queue:
        model_queue.append("llama-3.1-8b-instant")
    if "llama-3.3-70b-versatile" not in model_queue:
        model_queue.append("llama-3.3-70b-versatile")

    while iteration < max_iterations:
        iteration += 1
        response = None
        last_error = ""

        for candidate in model_queue:
            try:
                response = client.chat.completions.create(
                    model=candidate,
                    messages=scratchpad,
                    tools=AGENT_TOOLS,
                    tool_choice="auto",
                    max_tokens=1200
                )
                break
            except Exception as e:
                last_error = str(e)
                if status_callback and ("rate_limit" in last_error.lower() or "429" in last_error):
                    status_callback("warning", f"⚠️ {candidate} hit token limit. Shifting model...")
                continue

        if not response:
            return f"Model execution halted: {last_error}", None, None, time.perf_counter() - start_time

        msg = response.choices[0].message
        scratchpad.append(msg)

        # 2. Check for autonomous tool calls
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

    # 3. Clean Text Synthesis (Safe extraction from dicts vs Pydantic objects)
    if step_tool_count > 0 and (not final_text or len(final_text.strip()) == 0):
        if status_callback:
            status_callback("info", "Synthesizing executive briefing...")

        tool_outputs = []
        for m in scratchpad:
            # Handle both Python dicts and Groq Pydantic objects without throwing AttributeError
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
                    "Synthesize the inquiry and retrieved outputs into a clean executive briefing. "
                    "Use structured markdown tables, negative percentage drawdowns, and institutional takeaways. "
                    "NEVER include Python scripts, code blocks, HTML tags, or LaTeX formatting."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Inquiry: {user_input}\n\n"
                    f"Computed Data & Model Results:\n" + "\n\n".join(tool_outputs) + "\n\n"
                    "Deliver the structured briefing now."
                )
            }
        ]

        for candidate in model_queue:
            try:
                synth_resp = client.chat.completions.create(
                    model=candidate,
                    messages=synth_messages,
                    max_tokens=1500
                )
                final_text = synth_resp.choices[0].message.content or "Analysis completed."
                break
            except Exception as e:
                final_text = f"Synthesis note: {e}"

    elapsed = time.perf_counter() - start_time
    return final_text, generated_plot, generated_excel, elapsed
