# How I Approached the Problem

1. **Start from the analyst workflow**  
   Instead of simply asking an LLM to summarize a document, I modeled the process after what a human equity research associate does:
   - Read the entire PDF.
   - Extract headline figures (revenue, net income, EPS).
   - Return to tables for segment-level details.
   - Only use numbers explicitly written in the document.

2. **Define a strict schema**  
   I created a unified JSON schema that outlines every expected field:
   - Core financials.
   - Income statement / balance sheet / cash flow line items.
   - Key metrics and guidance.
   This ensures consistent structure for Citi, Tesla, or any other company.

3. **Multi‑pass extraction for accuracy**  
   I structured the pipeline into separate extraction passes:
   - **Base pass:** full-document extraction to capture headline metrics.
   - **Backfill pass:** only fills nulls, never overwrites existing values.
   - **Segment passes:** page‑level extraction of business segments  
     (e.g., Citi: Services, Markets; Tesla: Automotive, Energy, Services).

4. **Prevent hallucinations**  
   My prompts explicitly enforce:
   - No guessing.
   - Only extract numbers explicitly present.
   - Leave fields null if they do not appear.
   This keeps financial data trustworthy.

5. **Template + deep merge strategy**  
   I use:
   - A **template** with all required fields.
   - A **deep merge** that preserves non‑null values.
   This prevents the model from wiping out valid metrics during backfill.

6. **Segment inference logic**  
   I added helper functions that infer segments:
   - Citi: `_infer_citi_segment_from_label`
   - Tesla: `_infer_tsla_segment_from_label`
   If the LLM forgets to include a segment name but the label contains clues, the script assigns the correct segment.

7. **Clean Excel output**  
   I flattened JSON objects into tidy tabular rows:
   - One row per metric.
   - Clear columns for Category, Basis, Segment, Value, Unit.
   - QA logs highlight missing units.
   This creates an analyst‑ready Excel workbook.

8. **Iterative improvement strategy**  
   I planned for iterative refinements:
   - Improving prompts.
   - Adding more segment-level extractors.
   - Enhancing keyword-based table detection.
   The pipeline is designed to evolve and become more accurate over time.