import io
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sympy
from sympy import symbols, Symbol, Eq, solve, simplify, Rational
import yfinance as yf

def clean_code_snippet(code: str) -> str:
    cleaned_lines = []
    for line in code.strip().splitlines():
        trimmed = line.strip()
        if trimmed.startswith("```") or "plt.show()" in trimmed:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

def tool_python_interpreter(code: str, hitl_enabled: bool = True) -> tuple[str, bytes | None, bytes | None]:
    code = clean_code_snippet(code)
    
    if hitl_enabled:
        forbidden = ["import os", "import subprocess", "import shutil", "__import__", "open("]
        for term in forbidden:
            if term in code:
                return f"🛡️ HITL Intercept: '{term}' restricted.", None, None

    stdout_trap = io.StringIO()
    sys_stdout_backup = sys.stdout
    plot_bytes = None
    excel_bytes = None
    
    scope = {
        "math": math, "sympy": sympy, "symbols": symbols, "Symbol": Symbol,
        "Eq": Eq, "solve": solve, "simplify": simplify, "Rational": Rational,
        "pi": math.pi, "plt": plt, "np": np, "pd": pd, "yf": yf, "__builtins__": __builtins__
    }

    try:
        plt.close('all')
        sys.stdout = stdout_trap
        exec(
            "import math\nimport sympy\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n" + code,
            scope
        )
        
        # Capture figure buffer if plotted
        if plt.get_fignums():
            fig_buf = io.BytesIO()
            plt.gcf().savefig(fig_buf, format="png", bbox_inches="tight", dpi=140)
            fig_buf.seek(0)
            plot_bytes = fig_buf.getvalue()
            plt.close('all')

        # Capture DataFrame and auto-format Excel export
        df_representation = ""
        for var_name, var_val in scope.items():
            if isinstance(var_val, pd.DataFrame) and not var_name.startswith("_"):
                x_buf = io.BytesIO()
                with pd.ExcelWriter(x_buf, engine='openpyxl') as writer:
                    var_val.to_excel(writer, index=True, sheet_name="Data_Model")
                    worksheet = writer.sheets["Data_Model"]
                    for col in worksheet.columns:
                        max_len = max(len(str(cell.value or '')) for cell in col)
                        col_letter = col[0].column_letter
                        worksheet.column_dimensions[col_letter].width = max(max_len + 3, 12)
                x_buf.seek(0)
                excel_bytes = x_buf.getvalue()
                
                # Pass exact DataFrame values to prevent LLM calculation drift
                df_representation = f"\n\n[Exported DataFrame `{var_name}`]:\n" + var_val.to_string()
                break

        output = stdout_trap.getvalue().strip()
        if not output and df_representation:
            output = "Calculations executed successfully." + df_representation
        elif output and df_representation and df_representation not in output:
            output += df_representation
        elif not output:
            output = "Calculations executed successfully."

        return output, plot_bytes, excel_bytes

    except Exception as e:
        plt.close('all')
        return f"Python Execution Error: {e}", None, None
    finally:
        sys.stdout = sys_stdout_backup
