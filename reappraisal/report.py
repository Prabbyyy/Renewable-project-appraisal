"""Excel workbook export for the appraisal."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .model import ModelResult


def _autosize(ws) -> None:
    for col in ws.columns:
        width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 40)


def export_excel(result: ModelResult, sens_irr: pd.DataFrame, sens_dscr: pd.DataFrame,
                 tornado: pd.DataFrame, mc_samples: pd.DataFrame, mc_summary: dict,
                 path: Path) -> Path:
    cfg = result.config
    assumptions = pd.DataFrame(
        [(k, v) for k, v in asdict(cfg).items()], columns=["Assumption", "Value"])
    summary = pd.DataFrame(
        [(k, v) for k, v in result.metrics.items()], columns=["Metric", "Value"])
    mc_rows = [(f"{col} | {stat}", val)
               for col, stats in mc_summary["percentiles"].items()
               for stat, val in stats.items()]
    mc_rows += list(mc_summary["risk"].items())
    mc_tbl = pd.DataFrame(mc_rows, columns=["Statistic", "Value"])

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        assumptions.to_excel(xw, "Assumptions", index=False)
        summary.to_excel(xw, "Summary", index=False)
        result.cashflow.to_excel(xw, "Cashflow", index=False)
        sens_irr.to_excel(xw, "Sens_EquityIRR")
        sens_dscr.to_excel(xw, "Sens_MinDSCR")
        tornado.to_excel(xw, "Tornado", index=False)
        mc_tbl.to_excel(xw, "MC_Summary", index=False)
        mc_samples.to_excel(xw, "MC_Samples", index=False)
        for ws in xw.book.worksheets:
            _autosize(ws)
            ws.freeze_panes = "A2"
    return path
