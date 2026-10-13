"""Deterministic DCF Valuation Engine with Input Validation."""
import io
import matplotlib.pyplot as plt
import pandas as pd
from oryn.models.schemas import PermissionCategory, ToolResult
from oryn.tools.registry import BaseTool


class DCFModelTool(BaseTool):
    name = "calculate_dcf"
    description = "Execute a 5-year DCF schedule, enterprise value calculation, chart, and spreadsheet."
    category = PermissionCategory.READ_ONLY

    def execute(
        self,
        base_revenue: float = 50.0,
        revenue_growth: float = 0.20,
        ebitda_margin: float = 0.25,
        wacc: float = 0.10,
        terminal_growth: float = 0.025,
        years: int = 5
    ) -> ToolResult:
        # Strict Gordon Growth Model check: terminal growth must be strictly lower than discount rate
        if terminal_growth >= wacc:
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid Gordon Growth: terminal growth ({terminal_growth:.1%}) must be strictly less than WACC ({wacc:.1%})."
            )

        years_idx = list(range(1, years + 1))
        rev = [base_revenue * ((1 + revenue_growth) ** t) for t in years_idx]
        ebitda = [r * ebitda_margin for r in rev]
        df = [1 / ((1 + wacc) ** t) for t in years_idx]
        pv_ufcf = [u * d for u, d in zip(ebitda, df)]
        sum_pv_ufcf = sum(pv_ufcf)

        terminal_value = (ebitda[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
        pv_terminal_value = terminal_value / ((1 + wacc) ** years)
        enterprise_value = sum_pv_ufcf + pv_terminal_value

        # Build clean tabular dataset with currency strictly isolated in headers
        df_proj = pd.DataFrame({
            "Year": [str(y) for y in years_idx],
            "Revenue (USD M)": [f"{x:.2f}" for x in rev],
            "EBITDA (USD M)": [f"{x:.2f}" for x in ebitda],
            "Discount Factor": [f"{x:.4f}" for x in df],
            "PV of UFCF (USD M)": [f"{x:.2f}" for x in pv_ufcf]
        })

        # Generate visual bar plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(['PV Cash Flows', 'PV Terminal Value'], [sum_pv_ufcf, pv_terminal_value], color=['#4a148c', '#7b1fa2'], width=0.45)
        ax.set_title(f'Enterprise Value: USD {enterprise_value:.2f}M', fontsize=12, fontweight='bold')
        plt.tight_layout()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=200)
        plt.close(fig)

        # Generate Excel workbook
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
            df_proj.to_excel(writer, sheet_name='DCF_Schedule', index=False)

        # Markdown representation without naked dollar signs in table cells
        rows = ["| " + " | ".join(map(str, row)) + " |" for row in df_proj.values]
        headers = "| " + " | ".join(df_proj.columns) + " |"
        dividers = "| " + " | ".join([":---"] * len(df_proj.columns)) + " |"
        table_md = "\n".join([headers, dividers] + rows)

        briefing = (
            f"### 📊 Institutional DCF Valuation Schedule\n\n"
            f"**Implied Enterprise Value:** **USD {enterprise_value:.2f}M**\n\n"
            f"{table_md}\n\n"
            f"* **Cumulative PV of Cash Flows:** USD {sum_pv_ufcf:.2f}M\n"
            f"* **Terminal Value (TV):** USD {terminal_value:.2f}M\n"
            f"* **Present Value of TV:** USD {pv_terminal_value:.2f}M\n"
            f"* **Implied Enterprise Value (EV):** **USD {enterprise_value:.2f}M**"
        )

        return ToolResult(
            success=True,
            data={
                "briefing": briefing,
                "plot_bytes": img_buf.getvalue(),
                "excel_bytes": excel_buf.getvalue(),
                "enterprise_value": enterprise_value
            },
            is_live=True
        )
