"""
data_loader.py
==============
Loads and normalises the raw Excel dataset.

The workbook has a single sheet with columns:
    A: State   B: Date   C: Total (sales)   D: Category

Dates are stored EITHER as Excel serial integers OR as 'DD-MM-YYYY' strings.
This module handles both formats transparently.
"""

from __future__ import annotations

import logging
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EXCEL_EPOCH = datetime(1899, 12, 30)          # Excel's date-zero


def _excel_serial_to_date(n: float) -> datetime:
    """Convert an Excel date serial number → Python datetime."""
    return _EXCEL_EPOCH + timedelta(days=int(n))


def _parse_cell_date(raw: str) -> Optional[datetime]:
    """
    Try to parse a date value that may be:
      - An Excel serial integer (stored as a numeric string, e.g. '43477')
      - A 'DD-MM-YYYY' string (stored in the shared-strings table)
    Returns None if parsing fails.
    """
    raw = raw.strip()

    # Try numeric (Excel serial)
    try:
        serial = float(raw)
        if 30000 < serial < 60000:          # plausible Excel date range
            return _excel_serial_to_date(serial)
    except ValueError:
        pass

    # Try 'DD-MM-YYYY'
    try:
        return datetime.strptime(raw, "%d-%m-%Y")
    except ValueError:
        pass

    # Try ISO 'YYYY-MM-DD'
    try:
        return datetime.strptime(raw, "%Y-%m-%d")
    except ValueError:
        pass

    # Try 'M/D/YYYY'
    try:
        return datetime.strptime(raw, "%m/%d/%Y")
    except ValueError:
        pass

    logger.warning("Could not parse date: %r", raw)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_excel(path: str | Path) -> pd.DataFrame:
    """
    Parse the forecasting case-study Excel file WITHOUT using openpyxl /
    pandas read_excel (they require the exact same packages).  We read the
    raw XML inside the .xlsx zip directly, which works with the stdlib alone.

    Returns
    -------
    pd.DataFrame with columns:
        state   : str   – US state name
        date    : datetime – observation date
        sales   : float  – weekly sales total
        category: str   – product category (always 'Beverages')
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset from %s", path)

    with zipfile.ZipFile(path) as zf:
        # --- shared strings (text values) ---
        ss_xml = zf.read("xl/sharedStrings.xml").decode("utf-8")
        shared_strings: list[str] = re.findall(r"<t[^>]*>([^<]*)</t>", ss_xml)

        # --- worksheet ---
        sh_xml = zf.read("xl/worksheets/sheet1.xml").decode("utf-8")

    # Parse all rows
    rows_raw = re.findall(
        r'<row r="(\d+)"[^>]*>(.*?)</row>', sh_xml, re.DOTALL
    )

    records: list[dict] = []
    skipped = 0

    for row_num_str, row_content in rows_raw:
        row_num = int(row_num_str)
        if row_num == 1:
            continue  # header

        # Each cell: <c r="A2" t="s"><v>3</v></c>  or  <c r="B2"><v>43477</v></c>
        cells = re.findall(
            r'<c r="([A-Z]+)\d+"(?:[^>]* t="([^"]*)")?[^>]*>'
            r'(?:<f>[^<]*</f>)?<v>([^<]*)</v>',
            row_content,
        )
        row: dict[str, str] = {}
        for col, ctype, val in cells:
            row[col] = shared_strings[int(val)] if ctype == "s" else val

        # Must have at least State, Date, Total
        if not ("A" in row and "B" in row and "C" in row):
            skipped += 1
            continue

        date = _parse_cell_date(row["B"])
        if date is None:
            skipped += 1
            continue

        try:
            sales = float(row["C"])
        except (ValueError, KeyError):
            skipped += 1
            continue

        records.append(
            {
                "state": row["A"].strip(),
                "date": date,
                "sales": sales,
                "category": row.get("D", "Beverages").strip(),
            }
        )

    if skipped:
        logger.warning("Skipped %d rows due to parse errors", skipped)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    logger.info(
        "Loaded %d rows | %d states | date range %s → %s",
        len(df),
        df["state"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


def get_state_series(df: pd.DataFrame, state: str) -> pd.Series:
    """
    Return a weekly pd.Series indexed by date for a single state.
    Multiple rows on the same date (different categories) are summed.
    """
    state_df = df[df["state"] == state].copy()
    state_df = state_df.groupby("date")["sales"].sum()
    state_df = state_df.resample("W-SAT").sum(min_count=1)   # Group by week, sum sales
    return state_df


def list_states(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique state names."""
    return sorted(df["state"].unique().tolist())
