# Sample data

- `synthetic_sustainability_report.txt` — text of a **synthetic** sustainability report for *Nordvik Components AB*,
  a **fictional company** made up to demonstrate the pipeline. None of its figures describe a real organisation.
- `build_sample_pdf.py` — renders the text to `nordvik_sustainability_report_2025.pdf` (5 pages):

  ```
  backend/.venv/Scripts/python sample_data/build_sample_pdf.py      # Windows
  backend/.venv/bin/python sample_data/build_sample_pdf.py          # macOS/Linux
  ```

The fixture contains: an energy reduction target (30% by 2030, 2020 baseline, owned by the COO), a diversity target
(40% women in management by 2027), implementation activities, multi-year performance data (a table and an inline
series), a vague unquantified claim (which must *not* count as improvement), a GRI "in accordance with" statement and
a limited-assurance statement. Data Security and Product Lifecycle deliberately lack performance data, to show the
*data gap* behaviour.

Expected result (with or without the LLM): Reporting 6; Energy Planning 5 / Execution 3 / Performance 2;
D&I Planning 5 / Execution 3 / Performance 2; Product Lifecycle Planning 4; Data Security Planning 4, Performance
"No evidence found".
