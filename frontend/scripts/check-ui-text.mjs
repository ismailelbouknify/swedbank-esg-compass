// UI text checks (run after `npm run build`; exit code 1 on any failure):
//  1. no technical AI details in the UI source or the built bundle
//  2. the old product name is gone
//  3. the Swedbank ESG Compass branding is present where it must be
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const TECHNICAL = [
  "llama_cpp", "llama.cpp", "model.gguf", ".gguf", "LLM_PROVIDER", "Llama", "BM25", "Local LLM",
  "n_threads", "llm_model", "llm_provider", "extraction_mode", "model_name", "llm_confidence", "triggered_rule",
  "tokens per second", "embedding", "inference",
];
// old product names (the phrase "ESG assessment" describing an assessment is still allowed)
const OLD_NAMES = ["ESG Assessment Platform", "AI Assessment Platform", "Local AI", ">ESG Assessment<", "\"ESG Assessment\""];

function files(dir, exts) {
  if (!existsSync(dir)) return [];
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) return files(p, exts);
    return exts.some((e) => p.endsWith(e)) ? [p] : [];
  });
}

const src = files("src", [".ts", ".tsx", ".css"]);
const dist = files("dist", [".js", ".html", ".css"]);
if (!dist.length) console.warn("dist/ not found: run `npm run build` first to also check the built bundle.");
const read = (f) => readFileSync(f, "utf8");
const failures = [];

for (const file of [...src, ...dist, "index.html"]) {
  const text = read(file);
  for (const term of TECHNICAL) if (text.toLowerCase().includes(term.toLowerCase())) failures.push(`${file}: technical term "${term}"`);
  for (const term of OLD_NAMES) if (text.includes(term)) failures.push(`${file}: old product name ${term}`);
}

function expect(ok, message) {
  if (!ok) failures.push(message);
}
const header = read("src/components/AppHeader.tsx");
const brand = read("src/brand.ts");
expect(brand.includes('PRODUCT_NAME = "Swedbank ESG Compass"'), "brand.ts must define PRODUCT_NAME = Swedbank ESG Compass");
expect(header.includes("PRODUCT_NAME") && /Swedbank\s*<\/span>ESG Compass/.test(header), "header must show Swedbank ESG Compass");
expect(brand.includes('LOGO_SRC = "/swedbank-logo.svg"') && !/https?:\/\//.test(brand + header), "logo must be the local /swedbank-logo.svg only");
expect(read("index.html").includes("<title>Swedbank ESG Compass</title>"), "index.html title must be Swedbank ESG Compass");
expect(read("src/pages/QuestionnaireExample.tsx").includes("Reference criteria used by Swedbank ESG Compass"),
  "questionnaire page must say: Reference criteria used by Swedbank ESG Compass");
if (dist.length) {
  const bundle = dist.map(read).join("\n");
  for (const s of ["Swedbank ESG Compass", "Evidence-based sustainability assessment", "Reference criteria used by Swedbank ESG Compass", "/swedbank-logo.svg"])
    expect(bundle.includes(s), `built bundle must contain "${s}"`);
  expect(read("dist/index.html").includes("<title>Swedbank ESG Compass</title>"), "built index.html title must be Swedbank ESG Compass");
}

if (failures.length) {
  failures.forEach((f) => console.error(`✗ ${f}`));
  process.exit(1);
}
console.log(`✓ ${src.length + dist.length + 1} UI files checked: branding present, old name gone, no technical AI details.`);
