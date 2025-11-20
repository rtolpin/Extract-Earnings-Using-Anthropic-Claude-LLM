"""
earnings_tsla_citi.py

Purpose:
  - Extract structured financial data from:
      1) Citi Q1 2025 earnings call transcript PDF
      2) Tesla Q2 2025 update deck PDF (slides-as-PDF)

  - Focus on the "standard" earnings-call data a human analyst would
    key into Excel: revenue, net income, EPS, margins, cash flow,
    key operational metrics, and guidance.

Design:
  - For each PDF, we run a full-document LLM extraction ("base" pass),
    then a second "backfill" pass that tries to fill in any remaining
    null values using the same document.
  - Single, unified JSON schema per document.
  - Multi-sheet Excel output:
        - Headline
        - All_Metrics
        - Guidance

Requires:
  pip install anthropic pypdf pandas openpyxl
  export ANTHROPIC_API_KEY="sk-ant-..."

Files this script expects:
  - "citi_earnings_q12025.pdf"
  - "TSLA-Q2-2025-Update.pdf"
"""

from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from pypdf import PdfReader
import pandas as pd


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Use whatever Claude model you actually have access to.
# You can swap this for your "claude-sonnet-4-5-20250929" if needed.
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"

MAX_OUTPUT_TOKENS = 2048
MAX_BACKFILL_TOKENS = 2048
MAX_ANTHROPIC_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

_api_key = "[PLACE YOUR API KEY HERE]"

_client = Anthropic(api_key=_api_key)


# ---------------------------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def pdf_to_full_text(path: str | Path) -> str:
    """
    Convert a multi-page PDF into a single clean text blob.
    """
    reader = PdfReader(str(path))
    chunks: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        chunks.append(txt)
    return _normalize_whitespace("\n\n".join(chunks))


def robust_json_extract(raw: str) -> Dict[str, Any]:
    """
    Try hard to extract JSON from an LLM response.

    Strategy:
      1) Direct json.loads
      2) Search for JSON object substring (balanced braces heuristic)
      3) Strip code fences and retry
    """
    raw = raw.strip()

    # 1) Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2) Look for JSON object substring
    candidates = re.findall(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", raw, flags=re.DOTALL)
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue

    # 3) Strip code fences
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Unable to parse JSON from model response.\n\nRAW:\n{raw}")


def _rescale_millions_to_billions_if_appropriate(
    value: Any,
    unit: Optional[str],
    label: Optional[str],
) -> tuple[Any, Optional[str]]:
    if not isinstance(value, (int, float)):
        return value, unit

    unit_clean = (unit or "").strip().lower()
    if unit_clean != "million":
        return value, unit

    if abs(value) < 1000:
        return value, unit

    text = (label or "").lower()
    financial_keywords = [
        "revenue",
        "net income",
        "income",
        "profit",
        "loss",
        "operating income",
        "cash flow",
        "cashflow",
        "free cash flow",
        "ebitda",
        "expenses",
        "expense",
        "capex",
    ]
    if not any(k in text for k in financial_keywords):
        return value, unit

    new_value = value / 1000.0
    new_unit = "billion"
    return new_value, new_unit



def _infer_citi_segment_from_label(label: Optional[str]) -> Optional[str]:
    """
    Best-effort mapping from a line-item label to a Citi segment name.
    This helps normalize things like:
      - "Services revenue"
      - "Markets total revenue"
      - "PBWM net income"
    into clean segment tags: "Services", "Markets", "PBWM", etc.
    """
    if not label:
        return None

    text = label.lower()

    # canonical segments you care about
    if "services" in text:
        return "Services"
    if "markets" in text:
        return "Markets"
    if "pbwm" in text or "personal banking and wealth management" in text:
        return "PBWM"
    if "personal banking" in text and "wealth" in text:
        return "PBWM"
    if "u.s. personal banking" in text or "us personal banking" in text:
        return "US Personal Banking"
    if "legacy franchises" in text:
        return "Legacy Franchises"

    # you can extend this with whatever else appears in the Citi tables
    return None


def _infer_tsla_segment_from_label(label: Optional[str]) -> Optional[str]:
    """
    Auto-detect Tesla segment based on label text.

    Recognizes:
      - Automotive
      - Energy
      - Services (incl. "Services and Other")
      - Anything that includes a clear segment name
    """
    if not label:
        return None

    text = label.lower()

    if "automotive" in text or "auto" in text:
        return "Automotive"

    if "energy" in text or "solar" in text or "storage" in text:
        return "Energy"

    if "services" in text or "service" in text or "other" in text:
        return "Services"

    return None


def _anthropic_json_call(
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> Dict[str, Any]:
    """
    Call Anthropic expecting STRICT JSON back.
    Retries with exponential backoff.
    """
    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_ANTHROPIC_RETRIES + 1):
        try:
            resp = _client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=max_output_tokens,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                }],
                temperature=0.0,
            )

            text_blocks = [b for b in resp.content if b.type == "text"]
            if not text_blocks:
                raise RuntimeError("LLM response contained no text blocks")

            raw = text_blocks[0].text.strip()
            return robust_json_extract(raw)

        except Exception as e:
            last_err = e
            print(f"[Anthropic] Error on attempt {attempt}/{MAX_ANTHROPIC_RETRIES}: {e}")
            if attempt == MAX_ANTHROPIC_RETRIES:
                break
            sleep_for = RETRY_BACKOFF_SECONDS * attempt
            print(f"[Anthropic] Backing off for {sleep_for:.1f}s...")
            time.sleep(sleep_for)

    if last_err:
        raise last_err
    raise RuntimeError("Unknown error in Anthropic call")


def _to_list(value) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


# ---------------------------------------------------------------------------
# UNIFIED EARNINGS SCHEMA + PROMPTS
# ---------------------------------------------------------------------------

# This "template" guarantees we always get the same structure, even if the
# model omits some keys. We merge model output into this.
EARNINGS_TEMPLATE = {
    "company_name": None,
    "ticker": None,
    "fiscal_period": None,
    "call_date": None,
    "currency": None,

    "core_financials": {
        "total_revenue": {"label": "Total revenue", "value": None, "unit": None, "yoy_growth_pct": None, "basis": None},
        "net_income": {"label": "Net income", "value": None, "unit": None, "yoy_growth_pct": None, "basis": None},
        "diluted_eps": {"label": "Diluted EPS", "value": None, "unit": "per_share", "yoy_growth_pct": None, "basis": None},
        "adjusted_eps": {"label": "Adjusted EPS", "value": None, "unit": "per_share", "yoy_growth_pct": None, "basis": None},
        "operating_income": {"label": "Operating income", "value": None, "unit": None, "yoy_growth_pct": None, "basis": None},
        "operating_margin_pct": None,
        "gross_margin_pct": None,
        "operating_cash_flow": {"label": "Operating cash flow", "value": None, "unit": None, "yoy_growth_pct": None, "basis": None},
        "free_cash_flow": {"label": "Free cash flow", "value": None, "unit": None, "yoy_growth_pct": None, "basis": None},
        "adjusted_ebitda": {"label": "Adjusted EBITDA", "value": None, "unit": None, "yoy_growth_pct": None, "basis": None},
    },

    "income_statement": [],
    "balance_sheet": [],
    "cash_flow": [],
    "key_metrics": [],
    "guidance": []
}

EARNINGS_SYSTEM_PROMPT = """
You are a senior sell-side equity research analyst.

You will receive the FULL TEXT of a single earnings document
(EITHER an earnings call transcript OR an earnings update deck).

Your job is to extract ALL standard financial and operational data exactly as an
analyst’s associate would manually key into Excel.

STRICT RULES:

- Use ONLY numbers explicitly stated in the document.
- Do NOT guess, estimate, interpolate, or infer anything.
- If a metric does NOT appear, set its value to null.
- You MUST normalize any shorthand units to exact words:
    - "B", "bn", "BN", "billion(s)" => "billion"
    - "M", "MM", "m", "mm", "million(s)" => "million"
    - "K", "k", "thousands" => "thousand"
- If a table or footnote says the numbers are "in millions" or "in $ millions",
  then:
    - keep the raw cell amount as-is (e.g. 22,496)
    - set "unit": "million"
- If a table or footnote says the numbers are "in billions", then:
    - keep the raw cell amount as-is (e.g. 22.5)
    - set "unit": "billion"
- For percentage metrics (margins, ROE, growth rates, etc.):
    - store the numeric percentage as a pure number (e.g. 17.2 for 17.2%)
    - set "unit": "percent"
- Currency:
    - If the document is clearly in US dollars, use "USD" for currency where applicable.
    - If no currency is explicitly stated, you may leave "currency" as null.

Return STRICT JSON only.
No commentary, no explanation, no prose.

You MUST conform exactly to the following JSON schema:

{
  "company_name": string or null,
  "ticker": string or null,
  "fiscal_period": string or null,
  "call_date": string or null,
  "currency": string or null,

  "core_financials": {
    "total_revenue": {
      "label": "Total revenue",
      "value": number or null,
      "unit": "million" | "billion" | null,
      "yoy_growth_pct": number or null,
      "basis": "GAAP" | "Non-GAAP" | null
    },
    "net_income": {
      "label": "Net income",
      "value": number or null,
      "unit": "million" | "billion" | null,
      "yoy_growth_pct": number or null,
      "basis": "GAAP" | "Non-GAAP" | null
    },
    "diluted_eps": {
      "label": "Diluted EPS",
      "value": number or null,
      "unit": "per_share" | null,
      "yoy_growth_pct": number or null,
      "basis": "GAAP" | "Non-GAAP" | null
    },
    "adjusted_eps": {
      "label": "Adjusted EPS",
      "value": number or null,
      "unit": "per_share" | null,
      "yoy_growth_pct": number or null,
      "basis": "GAAP" | "Non-GAAP" | null
    },
    "operating_income": {
      "label": "Operating income",
      "value": number or null,
      "unit": "million" | "billion" | null,
      "yoy_growth_pct": number or null,
      "basis": "GAAP" | "Non-GAAP" | null
    },
    "operating_margin_pct": number or null,
    "gross_margin_pct": number or null,
    "operating_cash_flow": {
      "label": "Operating cash flow",
      "value": number or null,
      "unit": "million" | "billion" | null,
      "yoy_growth_pct": number or null,
      "basis": "GAAP" | "Non-GAAP" | null
    },
    "free_cash_flow": {
      "label": "Free cash flow",
      "value": number or null,
      "unit": "million" | "billion" | null,
      "yoy_growth_pct": number or null,
      "basis": "GAAP" | "Non-GAAP" | null
    },
    "adjusted_ebitda": {
      "label": "Adjusted EBITDA",
      "value": number or null,
      "unit": "million" | "billion" | null,
      "yoy_growth_pct": number or null,
      "basis": "GAAP" | "Non-GAAP" | null
    }
  },

  "income_statement": [
    {
      "label": string,
      "value": number or null,
      "unit": "million" | "billion" | null,
      "yoy_growth_pct": number or null,
      "basis": "GAAP" | "Non-GAAP" | null,
      "segment": string or null
    }
  ],

  "balance_sheet": [
    {
      "label": string,
      "value": number or null,
      "unit": "million" | "billion" | null,
      "yoy_growth_pct": number or null,
      "basis": "GAAP" | "Non-GAAP" | null,
      "segment": string or null
    }
  ],

  "cash_flow": [
    {
      "label": string,
      "value": number or null,
      "unit": "million" | "billion" | null,
      "yoy_growth_pct": number or null,
      "basis": "GAAP" | "Non-GAAP" | null,
      "segment": string or null
    }
  ],

  "key_metrics": [
    {
      "label": string,
      "value": number or null,
      "unit": "million" | "billion" | "thousand" | "percent" | null,
      "yoy_growth_pct": number or null,
      "basis": "GAAP" | "Non-GAAP" | null,
      "segment": string or null
    }
  ],

  "guidance": [
    {
      "label": string,
      "metric": string or null,
      "period": string or null,
      "low": number or null,
      "high": number or null,
      "unit": "million" | "billion" | "percent" | null,
      "basis": "GAAP" | "Non-GAAP" | null
    }
  ]
}
"""

TSLA_SEGMENT_TABLE_SYSTEM_PROMPT = """
You are a senior equity analyst specializing in Tesla.

You will receive the text from ONE PAGE of Tesla's Q2 2025 Update Deck.
That page may contain:
  - Segment-level tables for Automotive, Energy, and Services (Services & Other)
  - A Financial Summary section with company-wide headline metrics.

Follow the SAME unit rules as in the main earnings schema:
  - "22.5B", "$22.5B", "22.5 bn"  → value = 22.5, unit = "billion"
  - "1,172M", "1.2M", "1.2 million" → unit = "million"
  - "4.1%" → value = 4.1, unit = "percent"
  - DO NOT guess units.

Return STRICT JSON:
{
  "income_statement": [ ... ],
  "key_metrics": [ ... ]
}
...
"""

BACKFILL_SYSTEM_PROMPT = """
You are a senior sell-side equity research analyst.

You will receive:
  1) The FULL TEXT of an earnings document.
  2) A PREVIOUS_JSON object that already follows the standardized earnings schema.

Your job in THIS PASS:

  - Fill in ONLY the fields where PREVIOUS_JSON has value = null.
  - NEVER overwrite any non-null values.
  - NEVER change metadata (company_name, ticker, fiscal_period, call_date, currency)
    unless the field is currently null.
  - NEVER invent numbers. Use ONLY numbers explicitly shown in the document.
  - When you cannot find the value in the text, LEAVE IT AS null.

UNIT RULES (MUST FOLLOW EXACTLY):

  - Normalize shorthand financial units:
        "B", "bn", "BN", "billion(s)"         → "billion"
        "M", "MM", "m", "mm", "million(s)"    → "million"
        "K", "k", "thousand(s)"               → "thousand"
  - If a table header says "in millions" or "in $ millions":
        * keep the numeric amount as shown in the table
        * set "unit": "million"
  - If a table header says "in billions":
        * keep the value as shown
        * set "unit": "billion"
  - Percentage metrics:
        * Store the numeric percentage as a pure number (e.g. 17.2 for 17.2%)
        * Set "unit": "percent"
  - EPS metrics:
        * ALWAYS use "unit": "per_share"
  - DO NOT guess units for financial metrics. If the document does not specify,
    leave the unit null.
  - Currency:
        * If the document is clearly in USD, use "USD".
        * If not explicitly stated, leave currency as null.

STRUCTURAL RULES:

  - Do NOT create duplicate metrics.
  - Do NOT add rows with all fields null.
  - Do NOT change the shape of the schema.
  - You MAY add new rows to:
        income_statement
        balance_sheet
        cash_flow
        key_metrics
        guidance
    IF they clearly appear in the text AND they fit the schema.

  - For each row you add:
        * Set the "label" field exactly as written in the document.
        * If the row belongs to a segment (e.g., "Services", "Automotive", etc.),
          set the segment name exactly as written. If no segment is shown, use null.

RETURN STRICT JSON ONLY.
NO commentary. NO prose. NO explanation.
"""

# ---------------------------------------------------------------------------
# EXTRACTION FUNCTIONS
# ---------------------------------------------------------------------------

def normalize_unit_for_metric(label: Optional[str], unit: Optional[str]) -> Optional[str]:
    """
    Normalize the 'unit' field for a metric to one of:
        'billion', 'million', 'thousand', 'percent', 'per_share', or None.

    We:
      1) Canonicalize common LLM outputs like "B", "bn", "MM", "pct", "%"
      2) Infer "per_share" and "percent" from the label if unit is missing
      3) NEVER guess scale (million vs billion) if there's no textual clue
    """

    # 0. Clean inputs
    if unit is not None:
        unit_clean = str(unit).strip().lower()
    else:
        unit_clean = None

    label_text = (label or "").strip().lower()

    # 1. Canonical mapping for common variants
    UNIT_MAP = {
        # billions
        "b": "billion",
        "bn": "billion",
        "billion": "billion",
        "billions": "billion",
        "usd billion": "billion",
        "usd billions": "billion",
        "$b": "billion",
        "$bn": "billion",

        # millions
        "m": "million",
        "mm": "million",
        "million": "million",
        "millions": "million",
        "usd million": "million",
        "usd millions": "million",
        "$m": "million",
        "$mm": "million",

        # thousands
        "k": "thousand",
        "thousand": "thousand",
        "thousands": "thousand",

        # percent
        "%": "percent",
        "pct": "percent",
        "percent": "percent",
        "percentage": "percent",

        # per share
        "per_share": "per_share",
        "per share": "per_share",
    }

    valid_units = {"billion", "million", "thousand", "percent", "per_share"}

    # If model gave a unit, try to normalize it
    if unit_clean:
        if unit_clean in UNIT_MAP:
            normalized = UNIT_MAP[unit_clean]
            if normalized in valid_units:
                return normalized
        # If it's already one of the valid ones, keep it
        if unit_clean in valid_units:
            return unit_clean
        # Anything else → drop it (we'll try to infer from label)
        unit_clean = None

    # 2. Infer per-share for EPS-style labels when unit missing
    if "eps" in label_text or "earnings per share" in label_text:
        return "per_share"

    # 3. Infer percent for margin / ROE / ratio labels
    if any(
        kw in label_text
        for kw in ["margin", "roe", "roa", "rotce", "ratio", "yield", "%"]
    ):
        return "percent"

    # 4. For pure volume metrics (units, deliveries, deployments), we DO NOT
    #    guess million vs thousand. The model must set it based on the text.
    volume_keywords = [
        "deliveries",
        "delivery",
        "units",
        "vehicles",
        "production",
        "produced",
        "deployed",
        "deployment",
        "mw",
        "mwh",
        "gw",
        "gwh",
    ]
    if any(kw in label_text for kw in volume_keywords):
        # If the model didn't specify, leave as None.
        return None

    # 5. For dollar-based P&L / cash flow metrics, we also do not guess scale.
    #    The model must interpret table headers like "in millions" vs "in billions".
    #    If the model didn't set a unit, keep None.
    financial_keywords = [
        "revenue",
        "net income",
        "income",
        "profit",
        "loss",
        "cash flow",
        "cashflow",
        "ebitda",
        "opex",
        "expenses",
        "expense",
        "capex",
        "operating income",
        "free cash flow",
        "gross profit",
    ]
    if any(kw in label_text for kw in financial_keywords):
        return None

    # 6. Fallback: no unit
    return None


def build_rows_from_company_block(
    key: str,
    data: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build structured rows for Excel.
    NO duplicates. NO double metrics. NO null-only rows.
    """
    headline_rows = []
    metric_rows = []
    guidance_rows = []

    if not isinstance(data, dict):
        return {"headline_rows": [], "metric_rows": [], "guidance_rows": []}

    company = data.get("company_name")
    ticker = data.get("ticker")
    fiscal_period = data.get("fiscal_period")
    call_date = data.get("call_date")
    currency = data.get("currency")
    source = data.get("source") or key

    # ---- CORE FINANCIALS ----
    core = data.get("core_financials") or {}

    for key_name, metric in core.items():
        # margins are scalar numbers
        if key_name in ("operating_margin_pct", "gross_margin_pct"):
            if metric is None:
                continue
            label = "Operating margin" if key_name == "operating_margin_pct" else "Gross margin"
            headline_rows.append({
                "Company": company,
                "Ticker": ticker,
                "Fiscal Period": fiscal_period,
                "Call Date": call_date,
                "Currency": currency,
                "Category": "Core Financials",
                "Metric Key": key_name,
                "Line Item": label,
                "Value": metric,
                "Unit": "percent",
                "YoY Growth %": None,
                "Basis": None,
                "Source": source,
            })
            continue

        if not isinstance(metric, dict):
            continue

        # Skip pure-null metrics
        if metric.get("value") is None and metric.get("yoy_growth_pct") is None:
            continue

        label = metric.get("label") or key_name.replace("_", " ").title()
        unit = normalize_unit_for_metric(label, metric.get("unit"))

        headline_rows.append({
            "Company": company,
            "Ticker": ticker,
            "Fiscal Period": fiscal_period,
            "Call Date": call_date,
            "Currency": currency,
            "Category": "Core Financials",
            "Metric Key": key_name,
            "Line Item": label,
            "Value": metric.get("value"),
            "Unit": unit,
            "YoY Growth %": metric.get("yoy_growth_pct"),
            "Basis": metric.get("basis"),
            "Source": source,
        })

    # ------------------------------------------------------------------
    # 2) INCOME STATEMENT / BALANCE SHEET / CASH FLOW / KEY METRICS
    # ------------------------------------------------------------------
    def add_block(block_name: str, rows: Any) -> None:
        if not isinstance(rows, list):
            return

        for m in rows:
            if not isinstance(m, dict):
                continue

            value = m.get("value")
            yoy = m.get("yoy_growth_pct")

            # Skip pure null rows
            if value is None and yoy is None:
                continue

            label = m.get("label")
            unit = normalize_unit_for_metric(label, m.get("unit"))

            metric_rows.append({
                "Company": company,
                "Ticker": ticker,
                "Fiscal Period": fiscal_period,
                "Call Date": call_date,
                "Currency": currency,
                "Category": block_name,           # "Income Statement", "Key Metrics", etc.
                "Line Item": label,
                "Value": value,
                "Unit": unit,
                "YoY Growth %": yoy,
                "Basis": m.get("basis"),
                "Segment": m.get("segment"),
                "Source": source,
            })

    add_block("Income Statement", data.get("income_statement"))
    add_block("Balance Sheet", data.get("balance_sheet"))
    add_block("Cash Flow", data.get("cash_flow"))
    add_block("Key Metrics", data.get("key_metrics"))

    # ------------------------------------------------------------------
    # 3) GUIDANCE
    # ------------------------------------------------------------------
    for g in data.get("guidance") or []:
        if not isinstance(g, dict):
            continue

        low = g.get("low")
        high = g.get("high")

        # skip guidance rows with no numeric range at all
        if low is None and high is None:
            continue

        guidance_rows.append({
            "Company": company,
            "Ticker": ticker,
            "Fiscal Period": fiscal_period,
            "Call Date": call_date,
            "Currency": currency,
            "Label": g.get("label"),
            "Metric": g.get("metric"),
            "Period": g.get("period"),
            "Low": low,
            "High": high,
            "Unit": g.get("unit"),
            "Basis": g.get("basis"),
            "Source": source,
        })

    # ------------------------------------------------------------------
    # 4) DEDUPLICATION (NO DOUBLE METRICS)
    # ------------------------------------------------------------------

    # Deduplicate headline rows by Metric Key (e.g. "total_revenue")
    dedup_headline: Dict[str, Dict[str, Any]] = {}
    for row in headline_rows:
        key_h = row.get("Metric Key") or row.get("Line Item")
        if key_h not in dedup_headline:
            dedup_headline[key_h] = row
        else:
            # prefer row with non-null Value / YoY
            existing = dedup_headline[key_h]
            if existing.get("Value") is None and row.get("Value") is not None:
                dedup_headline[key_h] = row
            elif (
                existing.get("Value") is not None
                and row.get("Value") is not None
                and existing.get("YoY Growth %") is None
                and row.get("YoY Growth %") is not None
            ):
                dedup_headline[key_h] = row

    headline_rows = list(dedup_headline.values())

    # Deduplicate metric rows by (Category, Line Item, Segment, Fiscal Period, Basis)
    dedup_metric: Dict[tuple, Dict[str, Any]] = {}
    for row in metric_rows:
        key_m = (
            row.get("Category"),
            row.get("Line Item"),
            row.get("Segment"),
            row.get("Fiscal Period"),
            row.get("Basis"),
        )
        existing = dedup_metric.get(key_m)
        if existing is None:
            dedup_metric[key_m] = row
        else:
            # choose the "better" row: more filled-in numeric fields
            v_existing = existing.get("Value")
            v_new = row.get("Value")
            yoy_existing = existing.get("YoY Growth %")
            yoy_new = row.get("YoY Growth %")

            def score(val, yoy):
                s = 0
                if val is not None:
                    s += 1
                if yoy is not None:
                    s += 1
                return s

            if score(v_new, yoy_new) > score(v_existing, yoy_existing):
                dedup_metric[key_m] = row

    metric_rows = list(dedup_metric.values())

    return {
        "headline_rows": headline_rows,
        "metric_rows": metric_rows,
        "guidance_rows": guidance_rows,
    }


def normalize_company_metadata(
    company_key: str,
    block: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Final cleanup step per company:
      - Fill in missing metadata (company_name, ticker, fiscal_period, call_date, currency)
        from hard-coded fallbacks for known docs.
      - Promote any legacy top-level metrics (total_revenue, net_income, etc.)
        into core_financials.
      - For Citi, rescale huge 'million' values to 'billion' where appropriate.
    """

    out = dict(block)

    # --- 1) Metadata fallbacks (unchanged) ---
    METADATA_FALLBACKS = {
        "citi_q1_2025": {
            "company_name": "Citigroup Inc.",
            "ticker": "C",
            "fiscal_period": "Q1 2025",
            "call_date": "2025-04-15",
            "currency": "USD",
        },
        "tsla_q2_2025": {
            "company_name": "Tesla, Inc.",
            "ticker": "TSLA",
            "fiscal_period": "Q2 2025",
            "call_date": "2025-07-23",  # deck webcast date
            "currency": "USD",
        },
    }

    fallback = METADATA_FALLBACKS.get(company_key, {})

    def set_if_missing(field: str):
        if out.get(field) is None and field in fallback:
            out[field] = fallback[field]

    for field in ["company_name", "ticker", "fiscal_period", "call_date", "currency"]:
        set_if_missing(field)

    # --- 2) Promote legacy metrics -> core_financials & dedupe (your existing helper) ---
    out = promote_legacy_headline_to_core(out)

    # --- 3) Citi-specific rescaling: big millions -> billions ---
    if company_key == "citi_q1_2025":
        core = out.get("core_financials") or {}
        for k, metric in core.items():
            if isinstance(metric, dict):
                val = metric.get("value")
                unit = metric.get("unit")
                label = metric.get("label") or k.replace("_", " ").title()
                new_val, new_unit = _rescale_millions_to_billions_if_appropriate(val, unit, label)
                metric["value"] = new_val
                metric["unit"] = new_unit
        out["core_financials"] = core

        # Also fix statement / metrics arrays
        for arr_name in ["income_statement", "cash_flow", "key_metrics", "balance_sheet"]:
            rows = out.get(arr_name)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                val = row.get("value")
                unit = row.get("unit")
                label = row.get("label")
                new_val, new_unit = _rescale_millions_to_billions_if_appropriate(val, unit, label)
                row["value"] = new_val
                row["unit"] = new_unit

    return out


def fix_citi_core_from_transcript(
    pdf_path: str | Path,
    block: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Citi-only override:
    - Parse the transcript text directly for core headline metrics
      (revenue, net income, EPS, RoTCE).
    - Only fill or correct values for citi_q1_2025.
    - This guarantees we get clean, properly-scaled 'billion' numbers.
    """
    text = pdf_to_full_text(pdf_path)
    out = deepcopy(block)

    core = out.get("core_financials") or {}
    if not isinstance(core, dict):
        core = {}
    out["core_financials"] = core

    # --- Helper to update one metric if missing or obviously wrong ---
    def set_metric(name: str, value: float, unit: str | None, basis: str | None = None):
        m = core.get(name) or {}
        # Only override if:
        #   - value is currently null, or
        #   - value is absurdly large in millions (e.g. 4100 million for 4.1 billion)
        curr_val = m.get("value")
        curr_unit = m.get("unit")
        too_big_millions = (
            curr_unit in ("million", "Million")
            and isinstance(curr_val, (int, float))
            and curr_val >= 1000
        )
        if curr_val is None or too_big_millions:
            m["label"] = m.get("label") or name.replace("_", " ").title()
            m["value"] = value
            m["unit"] = unit
            if basis is not None:
                m["basis"] = basis
            core[name] = m

    # --- Revenue: "...on $21.6 billion of revenues" ---
    m_rev = re.search(r"\$?([\d\.]+)\s*billion\s+of\s+revenue", text, flags=re.IGNORECASE)
    if m_rev:
        rev_val = float(m_rev.group(1))
        set_metric("total_revenue", rev_val, "billion", "GAAP")

    # --- Net income: "net income of $4.1 billion" ---
    m_ni = re.search(r"net income of\s+\$?([\d\.]+)\s*billion", text, flags=re.IGNORECASE)
    if m_ni:
        ni_val = float(m_ni.group(1))
        set_metric("net_income", ni_val, "billion", "GAAP")

    # --- EPS: "earnings per share of $1.96" ---
    m_eps = re.search(r"earnings per share of\s+\$?([\d\.]+)", text, flags=re.IGNORECASE)
    if m_eps:
        eps_val = float(m_eps.group(1))
        set_metric("diluted_eps", eps_val, "per_share", "GAAP")

    # You could also add RoTCE as a key_metrics row if you want:
    # "with an RoTCE of 9.1%"
    m_rotce = re.search(r"rotce of\s+([\d\.]+)\s*%", text, flags=re.IGNORECASE)
    if m_rotce:
        rotce_val = float(m_rotce.group(1))
        # Append to key_metrics if not already present
        km = out.get("key_metrics") or []
        if not isinstance(km, list):
            km = []
        exists = any(
            isinstance(r, dict)
            and (r.get("label") or "").lower().startswith("rotce")
            for r in km
        )
        if not exists:
            km.append({
                "label": "RoTCE",
                "value": rotce_val,
                "unit": "percent",
                "yoy_growth_pct": None,
                "basis": None,
                "segment": None,
            })
        out["key_metrics"] = km

    return out


def deep_merge_non_null(
    original: Dict[str, Any],
    new: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge new into original, but NEVER overwrite a non-null scalar with null.
    This protects against the backfill pass accidentally blanking out values.

    Rules:
      - If original[key] is a dict and new[key] is dict -> recurse.
      - If it's a list -> use new's list (we assume later pass is richer).
      - If it's scalar:
          * if new is None -> keep original
          * else -> use new
    """
    result = deepcopy(original)

    for k, v_new in new.items():
        if k not in result:
            result[k] = v_new
            continue

        v_orig = result[k]

        if isinstance(v_orig, dict) and isinstance(v_new, dict):
            result[k] = deep_merge_non_null(v_orig, v_new)
        elif isinstance(v_orig, list) and isinstance(v_new, list):
            result[k] = v_new
        else:
            # Scalars
            if v_new is None:
                # keep original
                continue
            result[k] = v_new

    return result

def deep_merge_template(
    template: Dict[str, Any],
    model_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge model_json into template:
      - Ensure all expected keys exist.
      - For nested dicts, merge recursively.
      - For list fields, rely entirely on model_json if present.
    """
    result = deepcopy(template)

    for k, v in model_json.items():
        if k not in result:
            # Allow model to add extra keys (we just carry them along)
            result[k] = v
            continue

        if isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_template(result[k], v)
        else:
            # Scalar or list: override template
            result[k] = v

    return result
    

def promote_legacy_headline_to_core(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix-up helper:
    - If the model wrote metrics using the *old* schema
      (top-level total_revenue, net_income, etc.),
      copy those into core_financials when core_financials is null/empty.
    - Then remove the legacy top-level metrics so you don't have duplicates.
    """

    if not isinstance(block, dict):
        return block

    result = dict(block)  # shallow copy is fine, we only touch one level

    # Ensure core_financials exists and is a dict
    core = result.get("core_financials")
    if not isinstance(core, dict):
        core = {}
    result["core_financials"] = core

    # These are the metric keys we expect either in core_financials or legacy top-level
    metric_keys = [
        "total_revenue",
        "net_income",
        "diluted_eps",
        "adjusted_eps",
        "operating_income",
        "operating_cash_flow",
        "free_cash_flow",
        "adjusted_ebitda",
    ]
    margin_keys = ["operating_margin_pct", "gross_margin_pct"]

    # Reference template so we always have the right shape
    template_core = EARNINGS_TEMPLATE.get("core_financials", {})

    # 1) Copy dict metrics (total_revenue, net_income, etc.)
    for mkey in metric_keys:
        legacy_metric = result.get(mkey)
        core_metric = core.get(mkey)

        # If legacy metric isn't a dict, nothing to promote
        if not isinstance(legacy_metric, dict):
            continue

        # Start from template shape if needed
        if not isinstance(core_metric, dict):
            core_metric = deepcopy(template_core.get(mkey, {}))

        # If core has no value but legacy does, copy over fields
        if core_metric.get("value") is None and legacy_metric.get("value") is not None:
            for field in ("label", "value", "unit", "yoy_growth_pct", "basis"):
                if legacy_metric.get(field) is not None:
                    core_metric[field] = legacy_metric.get(field)

        # Make sure label is at least something sensible
        if not core_metric.get("label"):
            default_label = template_core.get(mkey, {}).get("label")
            core_metric["label"] = default_label or mkey.replace("_", " ").title()

        core[mkey] = core_metric

    # 2) Copy scalar margins (operating_margin_pct, gross_margin_pct)
    for mkey in margin_keys:
        legacy_val = result.get(mkey)
        # Only promote if core is missing and legacy has a value
        if core.get(mkey) is None and legacy_val is not None:
            core[mkey] = legacy_val

    # 3) Remove legacy top-level metrics so we don't duplicate in final JSON
    for mkey in metric_keys + margin_keys:
        if mkey in result:
            result.pop(mkey, None)

    # Write back normalized core
    result["core_financials"] = core
    return result


def extract_earnings_from_pdf_base(
    pdf_path: str | Path,
    document_description: str,
) -> Dict[str, Any]:
    """
    First-pass extraction using the EARNINGS_SYSTEM_PROMPT.
    Produces a structured JSON matching the new schema.
    """
    full_text = pdf_to_full_text(pdf_path)

    user_prompt = f"""
    The following text is the FULL content of an earnings document.
    
    Document description:
    {document_description}
    
    Use the JSON schema from the system prompt.
    Extract all metrics explicitly present in the text.
    
    FULL DOCUMENT TEXT:
    \"\"\"{full_text}\"\"\"
    """

    model_json = _anthropic_json_call(
        system_prompt=EARNINGS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    # 1st pass merge → fills template but allows model values
    merged = deep_merge_template(EARNINGS_TEMPLATE, model_json)

    return merged


def backfill_earnings_from_pdf(
    pdf_path: str | Path,
    previous_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Second pass — fills in any still-null metrics using backfill prompt.
    NEVER overwrites non-null values from previous_json.
    """
    full_text = pdf_to_full_text(pdf_path)

    user_prompt = f"""
    You are performing a QA pass.
    
    You receive:
      1) FULL earnings document text
      2) PREVIOUS_JSON (partial extraction)
    
    Rules:
      - DO NOT overwrite any non-null values in PREVIOUS_JSON
      - Fill in ONLY metrics where PREVIOUS_JSON has value=null
      - Add any missing income_statement, balance_sheet, cash_flow,
        key_metrics, or guidance items that clearly appear in tables.
    
    Return ONLY JSON that updates missing data.
    
    FULL DOCUMENT:
    \"\"\"{full_text}\"\"\"
    
    PREVIOUS_JSON (JSON only, no commentary):
    {json.dumps(previous_json)}
    """

    patch_json = _anthropic_json_call(
        system_prompt=BACKFILL_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_output_tokens=MAX_BACKFILL_TOKENS,
    )
    
    # deep merge: only apply non-null values from patch
    merged = deep_merge_non_null(previous_json, patch_json)
    
    # reapply full template to ensure all keys exist
    final = deep_merge_template(EARNINGS_TEMPLATE, merged)
    
    return final


def extract_citi_segment_tables(
    pdf_path: str | Path,
    previous_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Citi-specific *table-based* segment extraction.

    - Scans pages of the Citi Q1 2025 earnings transcript PDF for segment-related content.
    - For each candidate page, asks the LLM to return segment-level:
        • income_statement rows
        • key_metrics rows
    - Target segments include:
        • Services
        • Markets
        • PBWM (Personal Banking & Wealth Management)
        • US Personal Banking
        • Legacy Franchises
        • TTS (Treasury & Trade Solutions)
        • Securities Services
        • ICG (Institutional Clients Group)
        • etc. (everything inferable from labels)
    - Dedupes rows across pages before merging.
    """

    reader = PdfReader(str(pdf_path))
    page_texts: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        page_texts.append(_normalize_whitespace(txt))

    SEGMENT_KEYWORDS = [
        "services",
        "markets",
        "pbwm",
        "personal banking",
        "wealth management",
        "legacy franchises",
        "treasury and trade solutions",
        "tts",
        "securities services",
        "institutional clients group",
        "icg",
        "us personal banking",
        "segment results",
        "segment performance",
    ]

    candidate_indices: List[int] = []
    for i, text in enumerate(page_texts):
        lower = text.lower()
        if any(kw in lower for kw in SEGMENT_KEYWORDS):
            candidate_indices.append(i)

    if not candidate_indices:
        return previous_json

    income_segments: List[Dict[str, Any]] = []
    key_metrics_segments: List[Dict[str, Any]] = []

    # Local system prompt so it doesn't affect other docs
    citi_segment_table_system_prompt = """
    You are a senior sell-side bank analyst specializing in Citigroup.

    You will receive the text from ONE PAGE of a Citi earnings document.
    That page may contain segment-level tables or narrative describing
    segment performance.

    Your job for THIS PAGE:

    1) Extract segment-level metrics for major Citi segments, such as:
        - "Services"
        - "Markets"
        - "PBWM" (Personal Banking & Wealth Management)
        - "US Personal Banking"
        - "Legacy Franchises"
        - "Treasury and Trade Solutions" (TTS)
        - "Securities Services"
        - "Institutional Clients Group" (ICG)
        - any other clearly named business segment

        Typical metrics:
        - Segment revenue / net revenue
        - Segment expenses
        - Segment pretax income
        - Segment net income
        - Segment-level margins or returns (ROE/ROTCE)
        - Other clearly defined segment metrics

    2) Interpret units STRICTLY:
        - "21.6B", "$21.6B", "21.6 bn" → value = 21.6, unit = "billion"
        - "430M", "430 million" → value in millions, unit = "million"
        - Percentages like "17.2%" → value = 17.2, unit = "percent"
        - Do NOT guess units. Only assign units that are clearly indicated
            by B/M/% notation or table headers (e.g. "($ in billions)").
        - If a number is given but the scale is ambiguous, you may leave unit null.

    3) For each row, set the "segment" field to the segment name,
        e.g. "Services", "Markets", "PBWM", "US Personal Banking", etc.
        If you cannot confidently identify a segment → leave "segment" as null.

    4) Only extract metrics that are explicitly present on THIS PAGE.
        Do NOT invent numbers.

    Return STRICT JSON ONLY with this structure:

    {
    "income_statement": [
        {
        "label": string,
        "value": number or null,
        "unit": "billion" | "million" | "percent" | null,
        "currency": string or null,
        "period": string or null,
        "yoy_growth_pct": number or null,
        "basis": "GAAP" | "Non-GAAP" | null,
        "segment": string or null
        }
    ],
    "key_metrics": [
        {
        "label": string,
        "value": number or null,
        "unit": "billion" | "million" | "thousand" | "percent" | null,
        "currency": string or null,
        "period": string or null,
        "yoy_growth_pct": number or null,
        "basis": "GAAP" | "Non-GAAP" | null,
        "segment": string or null
        }
    ]
    }

    Do NOT return any other top-level keys.
    Do NOT wrap in markdown.
    """

    for idx in candidate_indices:
        page_text = page_texts[idx]

        user_prompt = f"""
        You are analyzing ONE PAGE of Citi's Q1 2025 earnings document.

        PAGE INDEX: {idx + 1}

        PAGE TEXT:
        \"\"\"{page_text}\"\"\"
        """

        try:
            page_json = _anthropic_json_call(
                system_prompt=citi_segment_table_system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=MAX_BACKFILL_TOKENS,
            )
        except Exception as e:
            print(f"[CITI SEGMENTS] Failed on page {idx + 1}: {e}")
            continue

        # ---- income_statement rows from this page ----
        for row in page_json.get("income_statement") or []:
            if not isinstance(row, dict):
                continue

            label = row.get("label")
            if not label:
                continue

            seg = row.get("segment")
            if not seg:
                inferred = _infer_citi_segment_from_label(label)
                if inferred:
                    row["segment"] = inferred
                    seg = inferred

            if not seg:
                # For a segment-specific pass, skip rows where we
                # cannot clearly tie the metric to a segment.
                continue

            if row.get("value") is None and row.get("yoy_growth_pct") is None:
                continue

            income_segments.append(row)

        # ---- key_metrics rows from this page ----
        for row in page_json.get("key_metrics") or []:
            if not isinstance(row, dict):
                continue

            label = row.get("label")
            if not label:
                continue

            seg = row.get("segment")
            if not seg:
                inferred = _infer_citi_segment_from_label(label)
                if inferred:
                    row["segment"] = inferred
                    seg = inferred

            if not seg:
                continue

            if row.get("value") is None and row.get("yoy_growth_pct") is None:
                continue

            key_metrics_segments.append(row)

    # ---------- Small dedupe across pages ----------
    def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        dedup: Dict[tuple, Dict[str, Any]] = {}

        def score(r: Dict[str, Any]) -> int:
            s = 0
            if r.get("value") is not None:
                s += 1
            if r.get("yoy_growth_pct") is not None:
                s += 1
            if r.get("basis") is not None:
                s += 1
            if r.get("unit") is not None:
                s += 1
            return s

        for r in rows:
            key_tuple = (
                (r.get("label") or "").strip().lower(),
                r.get("segment"),
                r.get("period"),
                r.get("basis"),
                r.get("unit"),
            )
            existing = dedup.get(key_tuple)
            if existing is None:
                dedup[key_tuple] = r
            else:
                if score(r) > score(existing):
                    dedup[key_tuple] = r

        return list(dedup.values())

    income_segments = _dedupe_rows(income_segments)
    key_metrics_segments = _dedupe_rows(key_metrics_segments)

    if not income_segments and not key_metrics_segments:
        return previous_json

    merged = deepcopy(previous_json)

    existing_income = merged.get("income_statement") or []
    if not isinstance(existing_income, list):
        existing_income = []
    merged["income_statement"] = existing_income + income_segments

    existing_key = merged.get("key_metrics") or []
    if not isinstance(existing_key, list):
        existing_key = []
    merged["key_metrics"] = existing_key + key_metrics_segments

    merged = deep_merge_template(EARNINGS_TEMPLATE, merged)
    return merged


def extract_tsla_segment_tables(
    pdf_path: str | Path,
    previous_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Tesla-specific *table-based + financial-summary* extraction.

    - Scans pages of the Q2 2025 Update Deck for:
        • Segment-related content (Automotive / Energy / Services)
        • The main Financial Summary slide (total revenue, operating income,
          operating margin, cash, etc.)

    - For each candidate page, asks the LLM (via TSLA_SEGMENT_TABLE_SYSTEM_PROMPT)
      to return:
        • income_statement rows
        • key_metrics rows

    - Keeps:
        • Company-level Financial Summary rows (segment = None)
        • Segment rows with segment in {"Automotive", "Energy", "Services"}

    - Dedupes rows across pages before merging.
    """

    reader = PdfReader(str(pdf_path))
    page_texts: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        page_texts.append(_normalize_whitespace(txt))

    SEGMENT_KEYWORDS = [
        "automotive",
        "energy",
        "services",
        "services & other",
        "services and other",
        "by segment",
        "segment information",
        "segment reporting",
        "financial summary",
        "f i n a n c i a l s u m m a r y",
    ]

    candidate_indices: List[int] = []
    for i, text in enumerate(page_texts):
        lower = text.lower()
        if any(kw in lower for kw in SEGMENT_KEYWORDS):
            candidate_indices.append(i)

    if not candidate_indices:
        return previous_json

    income_segments: List[Dict[str, Any]] = []
    key_metrics_segments: List[Dict[str, Any]] = []

    for idx in candidate_indices:
        page_text = page_texts[idx]

        user_prompt = f"""
        You are analyzing ONE PAGE of Tesla's Q2 2025 Update Deck.

        This page may contain:
        • Segment-level tables for Automotive, Energy, and/or Services
        • AND/OR a Financial Summary section with company-wide headline metrics.

        Your job for THIS PAGE:

        1) Extract ALL explicitly stated numeric metrics that relate to:
            - Automotive segment
            - Energy segment
            - Services (Services & Other) segment
            - Company-wide Financial Summary for Q2 2025, including items such as:
                • Total revenue (e.g. "$22.5B", "22.5B")
                • Operating income (e.g. "$0.9B")
                • Operating margin (e.g. "4.1%")
                • Gross margin (if given)
                • Cash, cash equivalents & investments (e.g. "$36.8B")
                • Any other clearly defined Q2 2025 headline metric
                • Any segment-level revenue / gross profit / operating income

        2) Interpret units STRICTLY:
            - If a value is written like "22.5B", "$22.5B", "22.5 bn" → value = 22.5, unit = "billion"
            - If written like "1.2M", "1.2 million", "1,200M" → value in millions, unit = "million"
            - If written as a percentage like "4.1%" → value = 4.1, unit = "percent"
            - DO NOT guess units. Only assign units that are clearly implied by the notation.
            - If there is no B/M/MM/k/% indicator, you MAY leave unit as null.

        3) Segment handling:
            - For rows that clearly belong to a segment, set:
                    "segment": "Automotive" | "Energy" | "Services"
                Examples:
                    "Automotive revenue" → segment = "Automotive"
                    "Energy generation and storage revenue" → segment = "Energy"
                    "Services & other" → segment = "Services"
            - For company-wide Financial Summary metrics (e.g. total company revenue,
                operating income, cash balance), set:
                    "segment": null

        4) DO NOT invent numbers. Only use metrics explicitly present on THIS PAGE.

        Return STRICT JSON only, with exactly this top-level structure:

        {{
            "income_statement": [
                {{
                "label": string,
                "value": number or null,
                "unit": "billion" | "million" | "percent" | null,
                "currency": string or null,
                "period": string or null,
                "yoy_growth_pct": number or null,
                "basis": "GAAP" | "Non-GAAP" | null,
                "segment": "Automotive" | "Energy" | "Services" | null
                }}
            ],
            "key_metrics": [
                {{
                "label": string,
                "value": number or null,
                "unit": "billion" | "million" | "thousand" | "percent" | null,
                "currency": string or null,
                "period": string or null,
                "yoy_growth_pct": number or null,
                "basis": "GAAP" | "Non-GAAP" | null,
                "segment": "Automotive" | "Energy" | "Services" | null
                }}
            ]
        }}

        Do NOT return any other top-level keys.
        Do NOT wrap in markdown.

        PAGE INDEX: {idx + 1}

        PAGE TEXT:
        \"\"\"{page_text}\"\"\"
        """

        try:
            page_json = _anthropic_json_call(
                system_prompt=TSLA_SEGMENT_TABLE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_output_tokens=MAX_BACKFILL_TOKENS,
            )
        except Exception as e:
            print(f"[TSLA SEGMENTS] Failed on page {idx + 1}: {e}")
            continue

        # ---- income_statement rows from this page ----
        for row in page_json.get("income_statement") or []:
            if not isinstance(row, dict):
                continue

            label = row.get("label")
            if not label:
                continue

            # Normalize / infer segment if missing
            seg = row.get("segment")
            if not seg:
                inferred = _infer_tsla_segment_from_label(label)
                if inferred:
                    row["segment"] = inferred
                    seg = inferred

            # Company-wide summary rows: segment can be None → keep them
            if seg is not None and seg not in ("Automotive", "Energy", "Services"):
                continue

            if row.get("value") is None and row.get("yoy_growth_pct") is None:
                continue

            income_segments.append(row)

        # ---- key_metrics rows from this page ----
        for row in page_json.get("key_metrics") or []:
            if not isinstance(row, dict):
                continue

            label = row.get("label")
            if not label:
                continue

            seg = row.get("segment")
            if not seg:
                inferred = _infer_tsla_segment_from_label(label)
                if inferred:
                    row["segment"] = inferred
                    seg = inferred

            if seg is not None and seg not in ("Automotive", "Energy", "Services"):
                continue

            if row.get("value") is None and row.get("yoy_growth_pct") is None:
                continue

            key_metrics_segments.append(row)

    # ---------- Small dedupe across pages ----------
    def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        dedup: Dict[tuple, Dict[str, Any]] = {}

        def score(r: Dict[str, Any]) -> int:
            s = 0
            if r.get("value") is not None:
                s += 1
            if r.get("yoy_growth_pct") is not None:
                s += 1
            if r.get("basis") is not None:
                s += 1
            if r.get("unit") is not None:
                s += 1
            return s

        for r in rows:
            key_tuple = (
                (r.get("label") or "").strip().lower(),
                r.get("segment"),
                r.get("period"),
                r.get("basis"),
                r.get("unit"),
            )
            existing = dedup.get(key_tuple)
            if existing is None:
                dedup[key_tuple] = r
            else:
                if score(r) > score(existing):
                    dedup[key_tuple] = r

        return list(dedup.values())

    income_segments = _dedupe_rows(income_segments)
    key_metrics_segments = _dedupe_rows(key_metrics_segments)

    if not income_segments and not key_metrics_segments:
        return previous_json

    merged = deepcopy(previous_json)

    existing_income = merged.get("income_statement") or []
    if not isinstance(existing_income, list):
        existing_income = []
    merged["income_statement"] = existing_income + income_segments

    existing_key = merged.get("key_metrics") or []
    if not isinstance(existing_key, list):
        existing_key = []
    merged["key_metrics"] = existing_key + key_metrics_segments

    merged = deep_merge_template(EARNINGS_TEMPLATE, merged)
    return merged

    
    
def write_multi_sheet_excel(
    output: Dict[str, Any],
    excel_path: str | Path,
) -> None:

    excel_path = Path(excel_path)

    all_headline = []
    all_metrics = []
    all_guidance = []

    # Convert JSON → row sets
    for key, data in output.items():
        if not isinstance(data, dict) or "error" in data:
            continue
        rows = build_rows_from_company_block(key, data)
        all_headline.extend(rows["headline_rows"])
        all_metrics.extend(rows["metric_rows"])
        all_guidance.extend(rows["guidance_rows"])

    headline_df = pd.DataFrame(all_headline)
    metrics_df = pd.DataFrame(all_metrics)
    guidance_df = pd.DataFrame(all_guidance)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        wrote = False

        if not metrics_df.empty:
            combined = pd.concat([headline_df, metrics_df], ignore_index=True)
            combined.to_excel(writer, sheet_name="All_Metrics", index=False)
            wrote = True

        if not headline_df.empty:
            headline_df.to_excel(writer, sheet_name="Key_Financials", index=False)
            wrote = True

        if not guidance_df.empty:
            guidance_df.to_excel(writer, sheet_name="Guidance", index=False)
            wrote = True

        if not wrote:
            pd.DataFrame({"Info": ["No extracted data"]}).to_excel(
                writer, sheet_name="Info", index=False
            )

    print(f"\n\n[WRITE] Excel saved → {excel_path}")
    print(f"Please Open the Excel File to View All Metrics and Key Financials: {excel_path}")



def main(): 
    CITI_PDF = "citi_earnings_q12025.pdf"
    TSLA_PDF = "TSLA-Q2-2025-Update.pdf"
    
    print("=== Extracting Citi Q1 2025 (transcript) ===")
    try:
        citi_base = extract_earnings_from_pdf_base(
            CITI_PDF,
            document_description=(
                "Citi (Citigroup Inc.) Q1 2025 earnings call TRANSCRIPT for a large U.S. bank (ticker C). "
                "Focus on headline firm-wide metrics spoken in the prepared remarks: "
                "total firm revenue, expenses, net income, EPS, RoTCE, CET1 ratio, tangible book value per share, "
                "firm-wide reserves, reserve-to-loan ratios, card NCL rates, card FICO mix, and key segment results "
                "for Services, Markets, Banking, Wealth, and U.S. Personal Banking. "
                "Do NOT fabricate line items that only appear in tables; only use numbers explicitly said aloud."
            ),
        )

        print("[CITI] Base extraction complete. Running backfill...")
        citi_backfilled = backfill_earnings_from_pdf(CITI_PDF, citi_base)
    
        print("[CITI] Running segment-level enrichment (Services, Markets, etc.)...")
        citi_with_segments = extract_citi_segment_tables(CITI_PDF, citi_backfilled)
        
        print("[CITI] Normalizing Data...")
        citi_fixed = fix_citi_core_from_transcript(CITI_PDF, citi_with_segments)
        citi_result = normalize_company_metadata("citi_q1_2025", citi_fixed)

        print("[CITI] Data Extracted Successfully.")
    
    except Exception as e:
        print(f"[MAIN] Failed to process Citi transcript: {e}")
        citi_result = {"error": str(e)}
    
    print("\n=== Extracting Tesla Q2 2025 (update deck) ===")
    try:
        tsla_base = extract_earnings_from_pdf_base(
            TSLA_PDF,
            document_description=(
                "Tesla, Inc. Q2 2025 Update Deck (slides-as-PDF, ticker TSLA). "
                "Include headline results (total revenue, gross profit, operating "
                "income, net income, EPS, OCF, FCF, adjusted EBITDA), key "
                "operational metrics (deliveries, production, energy, margins), "
                "and numeric guidance."
            ),
        )
        print("[TSLA] Base extraction complete. Running backfill...")
        tsla_backfilled = backfill_earnings_from_pdf(TSLA_PDF, tsla_base)

        print("[TSLA] Running segment-level enrichment (Automotive, Energy, Services, etc.)...")
        tsla_with_segments = extract_tsla_segment_tables(TSLA_PDF, tsla_backfilled)

        print("[TSLA] Normalizing Data...")
        tsla_result = normalize_company_metadata("tsla_q2_2025", tsla_with_segments)

        print("[TSLA] Data Extracted Successfully.")
        
    except Exception as e:
        print(f"[MAIN] Failed to process Tesla deck: {e}")
        tsla_result = {"error": str(e)}
    
    combined_output = {
        "citi_q1_2025": citi_result,
        "tsla_q2_2025": tsla_result,
    }
    
    print("\n=== COMBINED JSON OUTPUT ===")
    print(json.dumps(combined_output, indent=2))
    
    write_multi_sheet_excel(combined_output, "earnings_combined_output.xlsx")

#Runs Main Function to Execute Script
main()
