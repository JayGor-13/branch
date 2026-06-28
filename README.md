# BRANCH: Boosted Reasoning and Agentic Narratives for Clinical Healthcare

BRANCH wraps tabular clinical XGBoost predictors with SHAP attribution, DiCE
counterfactuals for classification tasks, and a clinical-document RAG guardrail.
The RAG corpus is an external folder of trusted clinical PDFs/TXT/MD documents,
not generated LLM summaries.

## Datasets

Supported paper datasets:

```text
gallstone        data/raw/Gallstone_disease/gallstone.csv
maternal_health  data/raw/maternal_health/*.csv
npha             data/raw/npha/NPHA-doctor-visits.csv
load_diabetes    sklearn.datasets.load_diabetes()
```

Gallstone, Maternal Health Risk, and NPHA are classification tasks. NPHA is
binarized as high utilization when the record is in the highest observed doctor
visit category. `load_diabetes` is the regression task.

## Clinical PDF Corpus

Place trusted clinical documents here:

```text
data/external/clinical_guidelines/pdfs/
```

Subfolders are optional, but useful:

```text
data/external/clinical_guidelines/pdfs/gallstone/
data/external/clinical_guidelines/pdfs/maternal_health/
data/external/clinical_guidelines/pdfs/npha/
data/external/clinical_guidelines/pdfs/diabetes/
```

The index builder accepts `.pdf`, `.txt`, and `.md` files.

## Run The Pipeline

Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

Preprocess and train all paper datasets:

```powershell
py .\scripts\preprocess_all.py --dataset all
py .\scripts\train_all_models.py --dataset all
```

Build the clinical document RAG index after adding PDFs:

```powershell
py .\scripts\build_guideline_index.py --pdf-dir data/external/clinical_guidelines/pdfs --dataset all --embedding-provider local
```

Generate SHAP and DiCE artifacts:

```powershell
py .\scripts\run_shap_all.py --dataset gallstone --limit 50
py .\scripts\run_shap_all.py --dataset maternal_health --limit 50
py .\scripts\run_shap_all.py --dataset npha --limit 50
py .\scripts\run_shap_all.py --dataset load_diabetes --limit 50

py .\scripts\run_dice_all.py --dataset gallstone --limit 20
py .\scripts\run_dice_all.py --dataset maternal_health --limit 20
py .\scripts\run_dice_all.py --dataset npha --limit 20
```

Run BRANCH explanations grounded in the clinical PDF index:

```powershell
py .\scripts\run_branch_explanations.py --dataset maternal_health --limit 10 --vector-index-path artifacts/vector_store/clinical_guidelines
```

Evaluate explanation quality using guideline retrieval:

```powershell
py .\scripts\evaluate_explanations.py --dataset maternal_health --vector-index-path artifacts/vector_store/clinical_guidelines
```

Generate paper-ready result tables under `results/tables/`:

```powershell
py .\scripts\build_seeded_guardrail_benchmark.py --datasets all --vector-index-path artifacts/vector_store/clinical_guidelines --embedding-provider local
py .\scripts\generate_paper_tables.py
```

For offline smoke tests without PDFs, the guideline builder can use the small
built-in maternal chunks:

```powershell
py .\scripts\build_guideline_index.py --use-default-curated --embedding-provider local
```

## LLM Narratives

The BRANCH agent can call any API or local model server that exposes an
OpenAI-compatible `/chat/completions` endpoint. Keep the provider as `template`
for deterministic narratives, or use `gemini` / `openai_compatible` for an LLM.

Example:

```powershell
$env:GEMINI_API_KEY="your_google_ai_studio_key"

py .\scripts\run_branch_explanations.py --dataset maternal_health --limit 10 --llm-provider gemini --llm-model gemma-4-31b-it --vector-index-path artifacts/vector_store/clinical_guidelines
```

If the API/server is unavailable, BRANCH falls back to the deterministic
template generator unless `--no-llm-fallback` is supplied.

## Table IV Gemini Variant Runs

For the two paper rows in Table IV, run BRANCH once per Gemini-backed variant.
The table labels remain:

```text
BRANCH-Gemma4-26B
BRANCH-Gemma4-31B
```

The actual Gemini API model ids can be overridden with environment variables.
The defaults are the two Gemma model ids used for the paper variants.

```powershell
$env:GEMINI_API_KEY="your_google_ai_studio_key"

$env:BRANCH_GEMMA4_26B_MODEL="gemma-4-26b-a4b-it"
$env:BRANCH_GEMMA4_31B_MODEL="gemma-4-31b-it"

python .\scripts\run_gemini_variants.py --datasets gallstone maternal_health npha --limit 20 --variants 26b 31b --resume --quality-mode ragas --vector-index-path artifacts/vector_store/clinical_guidelines --embedding-provider local --llm-request-delay-sec 5 --ragas-max-workers 1 --ragas-record-delay-sec 20
```

If a Gemini API run fails partway through with a transient 500/429 error, resume
only the missing model variant instead of rerunning everything:

```powershell
python .\scripts\run_gemini_variants.py --datasets gallstone maternal_health npha --limit 20 --variants 31b --resume --quality-mode ragas --vector-index-path artifacts/vector_store/clinical_guidelines --embedding-provider local --llm-request-delay-sec 5 --ragas-max-workers 1 --ragas-record-delay-sec 20
```

This writes one explanation-quality CSV per variant and dataset under
`results/metrics/`, then regenerates `results/tables/table_iv_explanation_quality.*`.
It also stores variant-specific BRANCH traces under
`artifacts/gemini_variants/` so Table V can compare 26B and 31B latency without
one run overwriting the other.
By default, explanation quality is computed with the external `ragas` package
using Gemini as the evaluator LLM.
Use `--allow-template-fallback` only for smoke tests; omit it for real paper
runs so failed Gemini calls do not become template narratives.

If Gemini returns a model-not-found error, first list the model ids available to
your API key:

```powershell
python .\scripts\run_gemini_variants.py --list-gemini-models
```

Then set `BRANCH_GEMMA4_26B_MODEL` and `BRANCH_GEMMA4_31B_MODEL` to real model
ids from that list. Names like `BRANCH-Gemma4-26B` are paper table labels; the
Gemini API needs actual model ids.

## Table V Latency Breakdown

Table V is generated from the variant-specific traces created by
`run_gemini_variants.py`. It reports latency only:

```text
XGBoost prediction
SHAP explanation
DiCE counterfactual
top-3 guideline RAG retrieval
clinical guardrail alignment
Gemini grounded summary generation
total end-to-end inference
```

After the 20-patient Gemini run, regenerate the tables with:

```powershell
python .\scripts\generate_paper_tables.py --table-iv-sample-size 20
```

To recompute only Table IV quality from existing traces/narratives with real
RAGAS, use resume mode. This skips already generated summaries but reruns the
RAGAS evaluation CSVs:

```powershell
python .\scripts\run_gemini_variants.py --datasets gallstone maternal_health npha --limit 20 --variants 26b 31b --resume --quality-mode ragas --vector-index-path artifacts/vector_store/clinical_guidelines --embedding-provider local --llm-request-delay-sec 5 --ragas-max-workers 1 --ragas-record-delay-sec 20
```

The paper table is written to:

```text
results/tables/table_v_guardrail_latency.tex
```

The detailed latency data is written to:

```text
results/tables/table_v_latency_per_patient.csv
results/tables/table_v_latency_by_dataset.csv
```
