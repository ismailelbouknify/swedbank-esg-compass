# Simple RAG vs baseline-v2.4 — fixed-document benchmark

Measured on 2026-11-25 on the same Windows laptop (8 logical CPUs, no GPU), with the same local model (`models/model.gguf`, Llama 3.2 1B Instruct, llama-cpp-python 0.3.35, 4096-token context), the same unchanged questionnaire (`esg_rules.json`, 16 questions) and the same two fixed PDF documents, uploaded as the only source (web search and web retrieval off, so no result depends on network availability). Each version ran in its own process with its own temporary database.

- **baseline-v2.4**: deterministic extraction + scoring engine (tag `baseline-v2.4`).
- **simple-rag original**: first RAG implementation; its response schema allowed `"recommended_level": null` with status `EVIDENCE_FOUND`.
- **simple-rag corrected**: the same, with the two-structure response schema (status, level and evidence coupled).

Documents: *Nordvik Components AB* sustainability report 2025 (fictional, 5 pages, `sample_data/`) and Sandvik AB Annual Report 2024 (public, 160 pages, from annualreport.sandvik). Assessment years 2025 and 2024.

## Measured results

| Document | Version | Answered with verified quotations | Unanswered | Level 1 without any quotation | Withheld for review | Quotations behind levels not in the document | Model calls | Model time (s) | Total time (s) | Peak memory (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| Nordvik | baseline-v2.4 | 8/16 (50%) | 8 | 7 | 0 | 0 | 10 | 160.3 | 160.8 ¹ | 1704.7 ¹ |
| Nordvik | simple-rag original | 4/16 (25%) | 12 | 0 | 12 | 0 | 28 | 595.2 | 595.7 | 1791.9 |
| Nordvik | simple-rag corrected | 11/16 (69%) | 5 | 0 | 5 | 0 | 25 | 529.2 | 529.6 | 1787.2 |
| Sandvik AB annual report 2024 | baseline-v2.4 | 3/16 (19%) | 13 | 12 | 0 | 0 | 26 | 578.2 | 601.2 | 1947.8 |
| Sandvik AB annual report 2024 | simple-rag original | 1/16 (6%) | 15 | 0 | 15 | 0 | 32 | 1247.0 | 1258.0 | 2108.0 |
| Sandvik AB annual report 2024 | simple-rag corrected | 7/16 (44%) | 9 | 0 | 9 | 0 | 28 | 907.4 | 917.7 | 2115.1 |

¹ The first baseline Nordvik run did not record memory; a second identical run measured 1704.7 MB peak and 131.2 s total (10 calls).

"Answered with verified quotations" counts levels the version itself presented as evidence-based (`FOUND`). baseline-v2.4 additionally assigned Level 1 to every question without evidence ("Level 1 without any quotation"), and some levels flagged for review; simple-rag never assigns a level without a verified quotation. Mean prompt size: baseline 1.9k (Nordvik) / 2.4k (Sandvik) characters per call; simple-rag 5.8k / 12.3k (one call answers a whole question from up to 7 passages). Every quotation was re-checked against the full PDF text independently of the versions' own verification.

## Answer correctness (reviewed, not independently)

Nordvik has documented expected results in `sample_data/README.md` (written with the fictional report by the project, not by an independent reviewer; 10 of the 16 questions). Sandvik has no reference scores. Agreement with the Nordvik reference (same level, or no level where the reference says "no evidence"):

- baseline-v2.4: 5/10
- simple-rag original: 1/10
- simple-rag corrected: 3/10

Every level the corrected simple-rag version recommended was reviewed manually against its quotation and the questionnaire criteria (S = the quotation supports the answer; W = a genuine quotation attached to a wrong or unsupported conclusion):

- Nordvik: 2 supported, 9 wrong or unsupported interpretations of genuine evidence.
- Sandvik: 2 supported, 5 wrong or unsupported interpretations of genuine evidence.

**Conclusion from these two documents.** The schema correction made the RAG pipeline functional (answers verified: Nordvik 4 → 11, Sandvik 1 → 7, with no invented quotation behind any level); like the original, it never assigns Level 1 for missing evidence. But with the 1B model most recommended levels rest on a genuine quotation that does not support them — typically a sentence about another factor, or a target read as activity or improvement. More questions answered is therefore **not** higher accuracy here: on the Nordvik reference the corrected RAG agreed on 3/10 questions against 5/10 for baseline-v2.4. The verification layer works as designed (no invented quotations reached a level; invented and misattributed quotations were withheld), and the semantic errors are exactly the documented limitation that is left to the analyst. These are two documents, not a representative benchmark; no architectural change was made on their basis.

## Per question — Nordvik (fictional sample, 5 pages)

| Question | Reference | baseline-v2.4 | simple-rag original | simple-rag corrected | Corrected: quotation (page) | Review |
|---|---|---|---|---|---|---|
| reporting:reporting | 6 | 6 | withheld | withheld |  |  |
| energy_management:planning | 5 | 5 (review) | withheld | withheld |  |  |
| energy_management:execution | 3 | 2 | withheld | 2 | "We aim to reduce energy consumption per unit produced by 30% by 2030 compared with a 2020 baseline." (p. 2) | W: cites the 30%-by-2030 target, not an activity |
| energy_management:performance | 2 | 2 | withheld | 2 | "We aim to reduce energy consumption per unit produced by 30% by 2030 compared with a 2020 baseline." (p. 2) | W: cites the target, not the intensity figures; level equals the reference by coincidence |
| product_lifecycle:planning | 4 | 1 (review) no quote | withheld | 3 | "Our sustainability strategy is approved by the Board of Directors and is reviewed annually by the Executive Co…" (p. 1) | W: general Board-approved sustainability strategy, not a product-lifecycle goal |
| product_lifecycle:execution | – | 1 (review) no quote | 2 | 2 | "In 2025 we launched a mentoring programme for female engineers and organised university recruitment events in …" (p. 3) | W: cites the mentoring programme for female engineers (another factor) |
| product_lifecycle:performance | – | 1 (review) no quote | withheld | 2 | "Energy use in our machining plants is our most material environmental impact. We aim to reduce energy consumpt…" (p. 2) | W: cites the energy target (another factor) |
| data_security:planning | 4 | 4 | 4 | 4 | "Our data security goal is to achieve and maintain ISO 27001 certification across all sites and to keep the num…" (p. 4) | S: ISO 27001 / zero-breach goal; level 4 equals the reference (explanation wrongly says 'ambitious') |
| data_security:execution | – | 1 (review) no quote | withheld | withheld |  |  |
| data_security:performance | no evidence | 1 (review) no quote | 2 | 2 | "Women in management: 2023: 28% / 2024: 31% / 2025: 34%" (p. 3) | W: cites the women-in-management table (another factor); reference: no evidence |
| labor_practices:planning | – | 1 (review) no quote | withheld | withheld |  |  |
| labor_practices:execution | – | 2 | withheld | withheld |  |  |
| labor_practices:performance | – | 1 (review) no quote | withheld | 2 | "Women in management: 2023: 28% / 2024: 31% / 2025: 34%" (p. 3) | W: cites the women-in-management table (another factor) |
| diversity_inclusion:planning | 5 | 4 | withheld | 4 | "Our goal is to reach 40% women in management positions by 2027." (p. 3) | S: 40% women in management by 2027; level 4 vs reference 5 |
| diversity_inclusion:execution | 3 | 2 | 2 | 2 | "In 2024 we installed 1.2 MW of rooftop solar panels at our Boras plant and replaced legacy air compressors to …" (p. 2) | W: cites the rooftop solar panels (another factor) |
| diversity_inclusion:performance | 2 | 2 | withheld | 2 | "In 2024 we installed 1.2 MW of rooftop solar panels at our Boras plant and replaced legacy air compressors to …" (p. 2) | W: cites the rooftop solar panels; level equals the reference by coincidence |

## Per question — Sandvik AB annual report 2024 (160 pages)

| Question | baseline-v2.4 | simple-rag original | simple-rag corrected | Corrected: quotation (page) | Review |
|---|---|---|---|---|---|
| reporting:reporting | 5 | withheld | 5 | "Sustainability is an integrated part of the reports submitted to the Audit Committee and sustainability report…" (p. 65) | W: quote is about reports to the Audit Committee; the GRI statement that supports L5 is not cited |
| energy_management:planning | 1 (review) no quote | withheld | 5 | "is essential to calculate and report Scope 3 GHG emissions." (p. 143) | W: 'is essential to calculate and report Scope 3 GHG emissions' does not show a goal |
| energy_management:execution | 1 (review) no quote | withheld | withheld |  |  |
| energy_management:performance | 1 | withheld | withheld |  |  |
| product_lifecycle:planning | 1 (review) no quote | withheld | 5 | "Our overarching objective is to achieve a waste circularity of 90 percent by 2030." (p. 145) | S: 'waste circularity of 90 percent by 2030': quantitative, timed goal |
| product_lifecycle:execution | 1 (review) no quote | withheld | 2 | "The enhanced tool provides data-driven insights into increased productivity, details the exact savings achieva…" (p. 42) | W?: customer savings tool; marginal as product-lifecycle activity |
| product_lifecycle:performance | 2 | 2 | 2 | "We also enhanced the overall circularity rate of our metal powders and made significant progress in our recond…" (p. 42) | S: 'enhanced the overall circularity rate of our metal powders' (qualitative) |
| data_security:planning | 1 (review) no quote | withheld | withheld |  |  |
| data_security:execution | 1 (review) no quote | withheld | withheld |  |  |
| data_security:performance | 1 (review) no quote | withheld | withheld |  |  |
| labor_practices:planning | 1 (review) no quote | withheld | withheld |  |  |
| labor_practices:execution | 1 (review) no quote | withheld | withheld |  |  |
| labor_practices:performance | 1 (review) no quote | withheld | withheld |  |  |
| diversity_inclusion:planning | 5 (review) | withheld | 5 | "Sandvik conducts regular Engagement Surveys (QPulse) to measure employee engagement and recognize improvement …" (p. 140) | W: engagement-survey sentence, not a goal (the 25%-female-managers goal exists, p139) |
| diversity_inclusion:execution | 1 (review) no quote | withheld | 2 | "We have zero tolerance for modern slavery in all its forms, forced labor, slave labor and/or child labor." (p. 67) | W: modern-slavery policy sentence (Labor Practices) |
| diversity_inclusion:performance | 1 (review) no quote | withheld | withheld |  |  |

## Diagnosis of the original simple-rag implementation

A rerun of three representative Nordvik questions with the model's raw responses logged locally showed that all 6 responses had `"status": "EVIDENCE_FOUND"` with `"recommended_level": null` — often naming the intended level in the explanation ("I recommend Level 4"). The original schema permitted `null` for the level whatever the status, so grammar-constrained decoding produced that combination and verification rejected it (`invalid_level`). The output was valid JSON; the schema did not couple status, level and evidence. The corrected schema offers exactly two structures (a supported answer with an integer level and 1–3 quotations, or an unsupported answer with `null` and no quotations); after the correction no `invalid_level` occurred. Remaining failures were genuine model errors that verification caught: quotations cited from the wrong chunk, lines merged with passage metadata, a changed first word, and the questionnaire's own level text copied as a "quotation". The model never chose `INSUFFICIENT_EVIDENCE` in the diagnostic, including for a question the report cannot answer.

Raw diagnostic logs were kept on the benchmark machine only and are not part of the repository.
