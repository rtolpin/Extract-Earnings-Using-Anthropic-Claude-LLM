Earnings Extraction Pipeline – Citi & Tesla

## Overview

This project automates what a typical equity research team (or buy-side analyst) does by hand after earnings:

- Read the Citi Q1 2025 earnings call transcript PDF

- Read the Tesla Q2 2025 update deck PDF

- Extract “standard” earnings call data:
 - Core financials: revenue, net income, EPS, cash flow, EBITDA, margins
 - Segment-level metrics (Citi & Tesla business lines)
 - Key metrics and guidance
 - `Write everything into a multi-sheet Excel file that a financial analyst can use directly.`

Design goals:
- High accuracy: use only numbers explicitly present in the PDFs.
- Minimal nulls: multi-pass extraction with a backfill step and segment-specific passes.
- Clean JSON → Excel mapping: no duplicate rows, clear units, and QA checks for missing units.
- The main entry point is the `main()` function at the bottom of `earnings_tsla_citi.py`.

## Prerequisites & Installation
Python version Recommended: Python 3.10+
(3.9+ will probably work, but you’re already on 3.11 in your notebook so that’s safe.)
Create and activate a virtual environment (recommended)

# From your project directory
`python -m venv venv` or `python3 -m venv venv`

# macOS / Linux
`source venv/bin/activate`

# Windows
`venv\Scripts\activate`

## Install required packages
`pip install anthropic pypdf pandas openpyxl`

Dependencies:

anthropic – to call Claude (your claude-sonnet-4-5-20250929 model)

pypdf – to read and extract text from the PDFs

pandas – to build DataFrames and write Excel

openpyxl – Excel writer engine for .xlsx

## Anthropic API key

In the ideal version, you set the key as an environment variable:

export ANTHROPIC_API_KEY="sk-ant-..."


And in code you would do:

_api_key = os.getenv("ANTHROPIC_API_KEY")
if not _api_key:
    raise RuntimeError("ANTHROPIC_API_KEY is not set...")
_client = Anthropic(api_key=_api_key)

`For now: replace _api_key = "[PLACE YOUR API KEY HERE]" with your Claude Anthropic API Key`

## Input PDFs

The script expects these files to exist in the same directory (or you can change the paths in `main()`):

`CITI_PDF = "citi_earnings_q12025.pdf"`
`TSLA_PDF = "TSLA-Q2-2025-Update.pdf"`

Place those PDFs next to earnings_tsla_citi.py.

## How to Run the Script

`replace _api_key = "[PLACE YOUR API KEY HERE]" with your Claude Anthropic API Key`

From the project directory (and with your venv activated):

`python earnings_tsla_citi.py` or `python3 earnings_tsla_citi.py`


## What you’ll see:

# Console logs like:

=== Extracting Citi Q1 2025 (transcript) ===

[CITI] Base extraction complete. Running backfill...

[TSLA] Running segment-level enrichment (Automotive, Energy, Services, etc.)...

The script prints the combined JSON for Citi and Tesla. And Generates an Excel File: earnings_combined_output.xlsx

## An Excel file is created: earnings_combined_output.xlsx

## with (at least) these sheets: 
# All_Metrics – all headline + segment metrics flattened into one table 
# Key_Financials – headline/core metrics for both companies

If there’s literally no data extracted, you’ll see a fallback Info sheet instead.

## Please Open the Excel File: earnings_combined_output.xlsx that is generated after the script has completed.
`open earnings_combined_output.xlsx`


## High-Level Design & Reasoning

# Problem framing

I framed the problem as:

“I have a team manually keying earnings call data into Excel. I want a script that gives me the same standard fields (revenue, net income, EPS, etc.), plus segment detail, with at least 90% accuracy.”

Key constraints I aimed for:

1. Single, consistent schema for all companies and documents.
2. High precision – no hallucinated numbers; only use what’s found in the PDFs.
3. Minimal nulls – LLM should try hard to fill values, but not invent data.
4. Segment-level detail – capture Services/Markets for Citi; Automotive/Energy/Services for Tesla.

# Overall pipeline

For each PDF, the pipeline is:

1. Read PDF
2. Convert pages to text with PdfReader.
3. Normalize whitespace (_normalize_whitespace).
4. Base extraction (full-document pass)
5. Use `EARNINGS_SYSTEM_PROMPT` and `extract_earnings_from_pdf_base()`.
6. Ask Claude to fill the entire unified schema: 
    - company_name, ticker, fiscal_period, call_date, currency
    - core_financials (total revenue, net income, EPS, etc.)
    - income_statement, balance_sheet, cash_flow, key_metrics, guidance
7. Merge into a template (EARNINGS_TEMPLATE) so all expected keys exist.
8. Backfill pass: `backfill_earnings_from_pdf()` runs a second LLM call with BACKFILL_SYSTEM_PROMPT.
9. The LLM receives:
    - Full document text
    - PREVIOUS_JSON (from base pass)
    - It is instructed to only fill nulls, not overwrite non-null values.
    - Merging uses a “non-null wins” strategy so existing values are never blanked out.
    - Segment-level table extraction
    - Separate targeted functions: `extract_citi_segment_tables()` for Citi and `extract_tsla_segment_tables()` for Tesla
        - These functions:
            1. Read each page’s text with PdfReader.
            2. Identify candidate pages (where segment tables likely appear) via keywords.
            3. For each candidate page: Call Claude with a segment-specific schema that only has: 
                - income_statement rows (with segment)
                - key_metrics rows (with segment)
            4. Clean and append those rows into the main income_statement / key_metrics arrays in your JSON.
    - Normalize metadata: `normalize_company_metadata()` ensures: company_name, ticker, fiscal_period, call_date, currency
            1. If the LLM didn’t extract these, you use a fallback mapping:
                - citi_q1_2025 → Citigroup Inc., C, Q1 2025, 2025-04-15, USD
                - tsla_q2_2025 → Tesla, Inc., TSLA, Q2 2025, 2025-07-23, USD
    - Flatten to rows Function: `build_rows_from_company_block()` transforms a structured company block into three lists of rows: 
            1. headline_rows, metric_rows, guidance_rows
            2. Each row has: 
                - Company, Ticker, Fiscal Period, Call Date, Currency
                - Category (e.g., Headline, Income Statement, Key Metrics)
                - Line Item, Value, Unit, YoY Growth %, Basis, Segment, Source
            3. It also: Normalizes units via normalize_unit_for_metric(...) (EPS → per_share, margins → percent).
            4. Skips “pure null” metrics (both value and yoy_growth_pct are None).
            5. Deduplicates: Headline metrics by metric key (e.g. total_revenue)
            6. Other metrics by (Category, Line Item, Segment, Fiscal Period, Basis)
10. Write Excel: `write_multi_sheet_excel()`:
  - Builds `headline_df`, `metrics_df`, `guidance_df` from all companies.
  - Writes: All_Metrics sheet = headline_df + metrics_df, Key_Financials sheet = headline_df, does not generate: Guidance sheet = guidance_df
  - If none of them have rows, writes a fallback Info sheet.

## Key Components & Logic

# Robust JSON extraction

`robust_json_extract(raw: str)`
- Tries direct json.loads.
- If that fails, uses a balanced braces regex to find a JSON object substring inside extra text.
- Strips ```json fences if the model wraps its output in a code block.
- Raises a helpful error if none of this works.
- This makes the script resilient even when the LLM occasionally wraps JSON in extra text.

# Template-based schema

`EARNINGS_TEMPLATE` defines a canonical shape:
 - Top-level metadata
 - core_financials (required core metrics)
 - Empty arrays for detailed statements / metrics / guidance

`deep_merge_template(template, model_json)`
- Recursively merges the model’s output into the template:
- Ensures all required keys exist.
- Lets the model fill in values.
- Allows extra keys from the model (you carry them through without breaking anything).

# Preserving non-null values

I used a merge function `deep_merge_non_null` that:
- Recursively walks both the previous JSON and the patch JSON.
- For scalars:
- If the patch has None: keep the original value.
- If the patch has a real value: overwrite.
- For dicts: Recurse and apply the same rule.
- For lists: Assume the patch’s list is better / newer and use it.
- This is what makes the backfill pass safe: it can never “wipe out” valid metrics.

# Segment inference helpers

`_infer_citi_segment_from_label(label)`:
- If the line item contains words like services, markets, PBWM, u.s. personal banking, etc., assign a canonical segment name.

`_infer_tsla_segment_from_label(label)`:
- If the label contains automotive, energy, services / services & other, etc., assign that segment.

- These are used when the LLM returns a valid metric but forgets to explicitly fill the segment field.
