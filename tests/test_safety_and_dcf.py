import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".streamlit"))

import agent
import sandbox
import tools


def test_python_interpreter_fails_closed():
    message, image_bytes, excel_bytes = sandbox.tool_python_interpreter("print('should not run')")
    assert "disabled" in message.lower()
    assert image_bytes is None
    assert excel_bytes is None


def test_unavailable_market_data_is_not_fabricated(monkeypatch):
    monkeypatch.setattr(tools, "fetch_yahoo_v8_chart", lambda symbol: None)
    monkeypatch.setattr(tools, "fetch_google_finance_metrics", lambda symbol: None)
    result = tools.tool_get_financial_data("TESTUNKNOWN")
    assert result["current_price"] == "N/A"
    assert result["market_cap"] == "N/A"
    assert result["data_status"] == "unavailable"
    assert result["source"] is None


def test_dcf_uses_assumptions_and_reports_model_defaults():
    report, plot_bytes, workbook_bytes = agent.generate_deterministic_dcf_artifacts(
        "DCF with starting revenue of $100 million, revenue growth 10%, "
        "EBITDA margin 20%, WACC 9%, terminal growth 2%"
    )
    assert "Year-0 starting revenue: USD 100.00M" in report
    assert "Annual revenue growth: 10.00%" in report
    assert "Illustrative" in report
    assert plot_bytes
    assert workbook_bytes


def test_dcf_rejects_invalid_terminal_growth():
    report, plot_bytes, workbook_bytes = agent.generate_deterministic_dcf_artifacts(
        "DCF with starting revenue 100 million, revenue growth 10%, "
        "EBITDA margin 20%, WACC 8%, terminal growth 9%"
    )
    assert "Invalid DCF assumptions" in report
    assert plot_bytes is None
    assert workbook_bytes is None

def test_formatter_preserves_fenced_code_blocks():
    source = "Example:\n\n```python\nprint('$5')\n```"
    assert agent.sanitize_report_formatting(source) == source


def test_dcf_audit_prompt_uses_ufcf_and_parses_year_one_revenue():
    prompt = (
        "Audit the DCF. Revenue: USD 60M in Year 1, growing 20% annually. "
        "EBITDA margin: apparently 25%. WACC: apparently 10%. "
        "Terminal growth: apparently 2.5%."
    )
    report, plot_bytes, workbook_bytes = agent.generate_deterministic_dcf_artifacts(prompt)
    assert "Revenue (USD M)" in report
    assert "UFCF (USD M)" in report
    assert "Cash Taxes (USD M)" in report
    assert "Change in NWC (USD M)" in report
    assert "Year-1 revenue of USD 60.00M" in report
    assert "USD 345.70M" not in report
    assert plot_bytes
    assert workbook_bytes
