# BRANCH: Boosted Reasoning and Agentic Narratives for Clinical Healthcare

## Detailed Paper Summary, Architecture, and Implementation Blueprint

**Project name:** BRANCH
**Full title:** Boosted Reasoning and Agentic Narratives for Clinical Healthcare
**Core domain:** Explainable clinical machine learning, tabular healthcare prediction, LLM-agent-based explanation generation
**Primary model:** XGBoost
**Explainability tools:** SHAP, DiCE counterfactuals
**Agentic layer:** ReAct-style LLM orchestration
**Safety layer:** Clinical guideline retrieval and anomaly checking
**Output:** Faithful, complete, clinically grounded natural-language explanation for each prediction

---

# 1. One-Line Summary

BRANCH is an agentic explainability framework that wraps a trained XGBoost clinical prediction model with SHAP-based attribution, DiCE-based counterfactual recourse, and a clinical guideline retrieval guardrail, then uses an LLM agent to synthesize these signals into a clear, faithful, and clinically aligned narrative explanation.

---

# 2. What This Paper Is Trying to Do

The paper is not mainly about creating a new predictive model. The predictive backbone is XGBoost, which is already a strong and practical choice for structured clinical/tabular data.

The real goal is to solve the following problem:

> Clinical machine learning models can produce strong predictions, but clinicians need to understand why a patient was assigned a risk score before trusting or acting on the output.

Traditional outputs such as feature importance plots, SHAP beeswarms, or counterfactual tables are useful to machine learning researchers but are often not directly usable by clinicians. They are technical, fragmented, and usually disconnected from clinical guidelines.

BRANCH tries to bridge this gap by converting raw model explanations into a clinical narrative.

The system should answer questions such as:

- Why was this patient flagged as high risk?
- Which features increased or decreased the model prediction?
- Are the model's reasons clinically sensible?
- What feature changes would move the prediction toward a lower-risk class?
- Is the model relying on a suspicious or medically contradictory correlation?

---

# 3. Core Research Claim

The main claim should be framed carefully:

> BRANCH preserves the predictive performance of XGBoost while improving the interpretability, completeness, clinical alignment, and actionability of its explanations compared with static SHAP or static DiCE outputs.

Important clarification:

- BRANCH should **not** be claimed to improve predictive performance unless the predictor itself is changed.
- BRANCH uses the same XGBoost model as the XGB baseline.
- Therefore, predictive metrics for `XGB-Base` and `BRANCH` should be identical or nearly identical.
- The improvement is in **explanation quality**, not raw accuracy.

---

# 4. High-Level System Architecture

BRANCH consists of three major layers:

```text
┌───────────────────────────────────────────────────────────────┐
│                     Layer 1: Orchestration                    │
│                                                               │
│  LLM Agent using ReAct-style tool orchestration               │
│  - Parses clinician query                                     │
│  - Calls predictor, SHAP, DiCE, and guideline tools           │
│  - Synthesizes final explanation                              │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                    Layer 2: Cognitive Tools                   │
│                                                               │
│  XGBoost Predictor                                            │
│  SHAP TreeExplainer                                           │
│  DiCE Counterfactual Generator                                │
│  Feature metadata and feasibility constraints                 │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                  Layer 3: Clinical Guardrail                  │
│                                                               │
│  Clinical guideline corpus                                    │
│  Embedding model                                              │
│  Vector database: FAISS / Chroma                              │
│  Guideline retrieval                                          │
│  Clinical alignment and anomaly detection                     │
└───────────────────────────────────────────────────────────────┘
```

---

# 5. Main Components

## 5.1 XGBoost Predictor

The XGBoost model is the prediction engine.

It receives a processed patient feature vector and outputs one of the following depending on the dataset:

- Binary disease probability
- Multi-class risk probability
- Multi-label disease/comorbidity prediction
- Continuous regression value

Example outputs:

```json
{
  "patient_id": 42,
  "task_type": "binary_classification",
  "predicted_class": "High Risk",
  "predicted_probability": 0.85
}
```

For regression:

```json
{
  "patient_id": 88,
  "task_type": "regression",
  "prediction": 8.0,
  "unit": "doctor_visits_per_year"
}
```

### Why XGBoost?

XGBoost is appropriate because:

- It performs strongly on tabular data.
- It handles non-linear feature interactions.
- It is robust on small-to-medium structured datasets.
- SHAP has efficient support for tree models through `TreeExplainer`.
- It is easier to interpret than deep neural networks for structured EHR-type data.

---

## 5.2 SHAP Explanation Tool

SHAP explains why the model made a prediction for a specific patient.

For a patient vector `x`, SHAP decomposes the model prediction into a base value plus feature-level contributions:

```math
hat{f}(x) = phi_0 + sum_{i=1}^{d} phi_i
```

Where:

- `φ₀` is the baseline prediction.
- `φᵢ` is the contribution of feature `i`.
- Positive SHAP values push risk upward.
- Negative SHAP values push risk downward.

### SHAP Tool Output Format

```json
{
  "patient_id": 42,
  "base_value": 0.41,
  "prediction": 0.85,
  "top_positive_features": [
    {
      "feature": "TotalCholesterol",
      "value": 245,
      "shap": 0.21,
      "direction": "increases_risk"
    },
    {
      "feature": "Age",
      "value": 64,
      "shap": 0.12,
      "direction": "increases_risk"
    }
  ],
  "top_negative_features": [
    {
      "feature": "PhysicalActivity",
      "value": "High",
      "shap": -0.05,
      "direction": "decreases_risk"
    }
  ]
}
```

### Role in BRANCH

SHAP is the evidence source for the LLM. The LLM should not invent causes. It should only narrate the features and directions returned by SHAP.

---

## 5.3 DiCE Counterfactual Tool

SHAP explains why a prediction happened. DiCE explains what would need to change to alter the prediction.

For example:

> If systolic blood pressure decreased from 150 to 135 and blood glucose normalized, the model would move from High Risk to Mid Risk.

### Counterfactual Objective

```math
min_{x'} d(x, x') quad text{s.t.} quad hat{f}(x') neq hat{f}(x), quad x' in mathcal{F}
```

Where:

- `x` is the original patient.
- `x'` is the counterfactual patient.
- `d(x, x')` is the distance between the original and counterfactual features.
- `F` is the feasibility constraint set.

### Feasibility Is Critical

Counterfactuals must be clinically reasonable.

Mutable features:

- BMI
- cholesterol
- blood pressure
- glucose
- sleep quality
- physical activity
- medication adherence, if available

Immutable or non-actionable features:

- age
- sex
- ethnicity
- past medical history, depending on context

DiCE should not suggest impossible changes such as:

- “Make the patient younger.”
- “Change biological sex.”
- “Remove historical diagnosis.”

### DiCE Tool Output Format

```json
{
  "patient_id": 17,
  "original_prediction": "High Risk",
  "counterfactual_prediction": "Mid Risk",
  "changes": [
    {
      "feature": "SystolicBP",
      "from": 150,
      "to": 135,
      "clinical_actionability": "potentially_modifiable"
    },
    {
      "feature": "BloodGlucose",
      "from": 8.5,
      "to": 6.9,
      "clinical_actionability": "potentially_modifiable"
    }
  ],
  "distance": 0.18,
  "feasibility_status": "valid"
}
```

---

## 5.4 Clinical Guardrail Layer

The clinical guardrail layer checks whether the model explanation is medically sensible.

It uses a vector-indexed clinical guideline corpus. The system retrieves relevant guideline chunks based on the top SHAP features and predicted clinical condition.

### Guardrail Inputs

```json
{
  "dataset": "MaternalHealth",
  "prediction": "High Risk",
  "top_shap_features": [
    {
      "feature": "SystolicBP",
      "direction": "increases_risk",
      "value": 150
    },
    {
      "feature": "BloodGlucose",
      "direction": "increases_risk",
      "value": 8.5
    }
  ]
}
```

### Guardrail Output

```json
{
  "guardrail_status": "no_anomaly",
  "retrieved_guidelines": [
    {
      "source": "ACOG",
      "topic": "Hypertension in pregnancy",
      "summary": "Elevated systolic blood pressure during pregnancy is associated with increased maternal risk.",
      "relevance_score": 0.87
    }
  ],
  "alignment_checks": [
    {
      "feature": "SystolicBP",
      "model_direction": "increases_risk",
      "clinical_direction": "increases_risk",
      "alignment": "concordant"
    }
  ]
}
```

### Anomaly Example

```json
{
  "guardrail_status": "anomaly_detected",
  "alignment_checks": [
    {
      "feature": "HDLCholesterol",
      "model_direction": "increases_risk",
      "clinical_direction": "decreases_or_protective",
      "alignment": "discordant"
    }
  ],
  "warning": "The model assigns a positive risk contribution to high HDL cholesterol, which may contradict standard cardiovascular interpretation. Treat prediction with caution."
}
```

### Guardrail Status Labels

Use a controlled set of labels:

```text
no_anomaly
possible_anomaly
anomaly_detected
insufficient_guideline_evidence
retrieval_failed
```

---

## 5.5 LLM ReAct Agent

The LLM agent is the orchestrator.

It should not make clinical claims from memory. It must rely on structured tool outputs.

### Agent Responsibilities

1. Parse the clinician query.
2. Identify the requested patient and task.
3. Call the XGBoost prediction tool.
4. Call the SHAP explanation tool.
5. Decide whether DiCE counterfactuals are needed.
6. Call the guideline retrieval tool.
7. Run clinical alignment check.
8. Generate final narrative.
9. Attach warning if anomaly is detected.

### Recommended Tool-Use Trace

Do not expose hidden chain-of-thought in the final application. Instead, log structured traces:

```json
{
  "query_id": "q_000042",
  "patient_id": 42,
  "tools_called": [
    "predict_xgboost",
    "explain_shap",
    "generate_counterfactual_dice",
    "retrieve_guidelines",
    "check_clinical_alignment",
    "generate_narrative"
  ],
  "guardrail_status": "no_anomaly",
  "final_output_id": "narrative_000042"
}
```

### LLM Prompting Rules

The LLM must follow these rules:

- Mention only features present in the SHAP output.
- Separate model evidence from clinical guideline evidence.
- Do not diagnose the patient.
- Do not recommend treatment unless supported by retrieved guideline evidence.
- Use uncertainty language: “the model suggests,” “the prediction is driven by,” “this may indicate.”
- Always include a guardrail status.
- If guideline evidence is missing, say so.
- If SHAP and guideline directions conflict, flag the explanation as clinically suspicious.

---

# 6. End-to-End BRANCH Workflow

## 6.1 Training-Time Workflow

```text
1. Load raw dataset
2. Clean missing values
3. Encode categorical variables
4. Standardize or normalize numeric features if needed
5. Split into train/test sets
6. Train baseline models
   - Logistic Regression / Ridge Regression
   - Random Forest
   - XGBoost
7. Tune XGBoost using Optuna
8. Save trained models
9. Evaluate predictive performance
10. Save predictions and metrics
```

---

## 6.2 Inference-Time Explanation Workflow

```text
Input:
  clinician query + patient ID

Step 1:
  Parse query and identify patient record

Step 2:
  Run trained XGBoost model

Step 3:
  Generate SHAP local explanation

Step 4:
  If patient is high-risk or user asks for actionability:
      Generate DiCE counterfactual
  Else:
      Skip DiCE or mark as optional

Step 5:
  Retrieve clinical guideline chunks using top SHAP features

Step 6:
  Compare model feature directions with guideline directions

Step 7:
  Generate final LLM narrative

Output:
  prediction + main drivers + counterfactual + guideline evidence + anomaly status
```

---

## 6.3 Pseudocode

```python
def branch_explain(query, patient_id, dataset_name):
    # 1. Load patient record
    x = load_patient_record(dataset_name, patient_id)

    # 2. Predict using XGBoost
    prediction = xgb_predict(dataset_name, x)

    # 3. Generate SHAP explanation
    shap_result = shap_explain(dataset_name, x)

    # 4. Generate counterfactual if needed
    if should_generate_counterfactual(query, prediction):
        dice_result = dice_counterfactual(dataset_name, x, prediction)
    else:
        dice_result = None

    # 5. Retrieve clinical guideline context
    guideline_context = retrieve_guidelines(
        dataset_name=dataset_name,
        prediction=prediction,
        shap_features=shap_result["top_features"]
    )

    # 6. Run clinical alignment check
    guardrail_result = check_alignment(
        shap_result=shap_result,
        guideline_context=guideline_context
    )

    # 7. Generate narrative
    narrative = generate_llm_narrative(
        query=query,
        prediction=prediction,
        shap_result=shap_result,
        dice_result=dice_result,
        guideline_context=guideline_context,
        guardrail_result=guardrail_result
    )

    # 8. Save trace
    save_explanation_trace(
        patient_id=patient_id,
        dataset_name=dataset_name,
        prediction=prediction,
        shap_result=shap_result,
        dice_result=dice_result,
        guideline_context=guideline_context,
        guardrail_result=guardrail_result,
        narrative=narrative
    )

    return narrative
```

---

# 7. Datasets

The draft currently includes four datasets.

| Dataset | Size | Features | Task | Purpose |
|---|---:|---:|---|---|
| Gallstone Disease | 320 | 37 | Binary classification | Disease risk prediction |
| Synthetic EHR | 10,000 | 11 | Multi-label / classification | Scalable synthetic EHR setting |
| Maternal Health Risk | 1,010 | 6 | 3-class classification | Clinically interpretable risk stratification |
| NPHA Doctor Visits | 714 | 15 | Regression | Healthcare utilization prediction |

## 7.1 Gallstone Disease

Target:

- Gallstone disease present or absent

Likely explanation features:

- Total cholesterol
- HDL
- LDL
- BMI
- Age
- Liver enzymes
- Physical activity

Expected output:

> Patient has high gallstone risk because cholesterol, BMI, and age push the model prediction upward.

---

## 7.2 Synthetic EHR

Target:

- Disease/comorbidity risk labels

Likely explanation features:

- Blood pressure
- BMI
- heart rate
- cholesterol
- glucose

Main value:

- Useful for demonstrating guardrail anomaly detection because synthetic data may contain spurious correlations.

---

## 7.3 Maternal Health Risk

Target:

- Low Risk
- Mid Risk
- High Risk

Features:

- Age
- SystolicBP
- DiastolicBP
- Blood sugar/glucose
- Body temperature
- Heart rate

Expected output:

> High risk is driven by elevated systolic blood pressure and blood glucose, with guideline support from pregnancy risk literature.

---

## 7.4 NPHA Doctor Visits

Target:

- Number of annual doctor visits

Task type:

- Regression

Expected output:

> The model predicts higher annual doctor visits due to prescription medication count, poor self-rated health, poor sleep quality, and mobility constraints.

---

# 8. Experimental Design

## 8.1 Predictive Baselines

Use the following baselines:

| Method | Purpose |
|---|---|
| Logistic Regression | Simple linear classification baseline |
| Ridge Regression | Regression baseline for NPHA |
| Random Forest | Nonlinear tree ensemble baseline |
| XGB-Base | Main predictive model |
| XGB+SHAP-Static | XGBoost with standard SHAP output only |
| XGB+DiCE-Static | XGBoost with standard DiCE output only |
| BRANCH | Full proposed framework |

Important:

- `XGB-Base`, `XGB+SHAP-Static`, `XGB+DiCE-Static`, and `BRANCH` should use the same trained XGBoost predictor.
- Their predictive metrics should match.
- Explanation metrics should differ.

---

## 8.2 Predictive Metrics

For binary and multi-class classification:

```text
Accuracy
Macro-F1
AUROC
Precision
Recall
Confusion matrix
```

For regression:

```text
MAE
RMSE
R2
```

---

## 8.3 Explanation Metrics

### Explanation Quality Score

```math
EQS = alpha F_{faith} + beta F_{comp} + gamma F_{align}
```

Recommended weights:

```text
alpha = 0.4
beta  = 0.3
gamma = 0.3
```

### Sub-Metrics

#### 1. Faithfulness

Measures whether the narrative matches SHAP importance.

Possible implementation:

```text
F_faith = rank_correlation(SHAP_feature_ranking, narrative_feature_ranking)
```

Simpler implementation:

```text
F_faith = number of correctly direction-matched cited features / number of cited features
```

#### 2. Completeness

Measures how many top SHAP features are mentioned.

```text
F_comp = number of top-k SHAP features mentioned / k
```

For example, if top-5 SHAP features are available and the narrative mentions 4:

```text
F_comp = 4 / 5 = 0.8
```

#### 3. Clinical Alignment

Measures whether the explanation agrees with guideline or expert judgment.

Possible scoring:

```text
1.0 = clinically concordant
0.5 = unclear / insufficient evidence
0.0 = clinically discordant
```

Alternatively, use binary expert review:

```text
1 = aligned
0 = not aligned
```

---

## 8.4 Guardrail Metrics

For deliberately seeded or detected anomaly cases:

```text
Guardrail Precision
Guardrail Recall
Guardrail F1
False Positive Rate
False Negative Rate
```

Definitions:

```text
Precision = correctly flagged anomalies / all flagged anomalies
Recall    = correctly flagged anomalies / all true anomalies
F1        = harmonic mean of precision and recall
```

---

## 8.5 Latency Metrics

Measure total explanation time and step-wise time.

```text
Total latency
Prediction latency
SHAP latency
DiCE latency
Guideline retrieval latency
LLM narrative generation latency
```

Save latency for every query.

Target:

```text
< 5 seconds per explanation if aiming for near-real-time clinical use
```

---

# 9. Output Narrative Structure

The final BRANCH explanation should follow a consistent template.

## 9.1 Classification Narrative Template

```text
Prediction:
The model predicts [CLASS] with [PROBABILITY]% confidence.

Main model drivers:
The largest upward contributors were [FEATURE_1], [FEATURE_2], and [FEATURE_3].
The largest protective or downward contributors were [FEATURE_4] and [FEATURE_5].

Clinical interpretation:
Retrieved guideline evidence indicates that [FEATURE_1] is clinically associated with [RISK_DIRECTION].

Counterfactual pathway:
The model would move toward [LOWER_RISK_CLASS] if [FEATURE_CHANGES], assuming other features remain fixed.

Guardrail status:
[NO_ANOMALY / POSSIBLE_ANOMALY / ANOMALY_DETECTED / INSUFFICIENT_EVIDENCE]

Caution:
This is a model explanation, not a diagnosis. Clinician review is required.
```

---

## 9.2 Regression Narrative Template

```text
Prediction:
The model predicts approximately [VALUE] [UNIT] for this patient.

Main model drivers:
The largest upward contributors were [FEATURE_1], [FEATURE_2], and [FEATURE_3].
The largest downward contributors were [FEATURE_4] and [FEATURE_5].

Clinical interpretation:
The retrieved evidence suggests that these factors are plausibly associated with increased healthcare utilization.

Counterfactual pathway:
Changing [FEATURE] from [OLD_VALUE] to [NEW_VALUE] would reduce the predicted value to approximately [COUNTERFACTUAL_VALUE].

Guardrail status:
[STATUS]

Caution:
This is a model explanation and should not replace clinical judgment.
```

---

# 10. Project Directory Structure

Recommended project layout:

```text
BRANCH/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── LICENSE
│
├── configs/
│   ├── default.yaml
│   ├── datasets/
│   │   ├── gallstone.yaml
│   │   ├── synthetic_ehr.yaml
│   │   ├── maternal_health.yaml
│   │   └── npha.yaml
│   │
│   ├── models/
│   │   ├── xgboost.yaml
│   │   ├── random_forest.yaml
│   │   └── logistic_regression.yaml
│   │
│   ├── branch/
│   │   ├── shap.yaml
│   │   ├── dice.yaml
│   │   ├── guardrail.yaml
│   │   └── llm_agent.yaml
│   │
│   └── experiments/
│       ├── exp_gallstone.yaml
│       ├── exp_synthetic_ehr.yaml
│       ├── exp_maternal_health.yaml
│       └── exp_npha.yaml
│
├── data/
│   ├── raw/
│   │   ├── gallstone/
│   │   ├── synthetic_ehr/
│   │   ├── maternal_health/
│   │   └── npha/
│   │
│   ├── interim/
│   │   ├── gallstone/
│   │   ├── synthetic_ehr/
│   │   ├── maternal_health/
│   │   └── npha/
│   │
│   ├── processed/
│   │   ├── gallstone/
│   │   │   ├── train.csv
│   │   │   ├── test.csv
│   │   │   ├── feature_metadata.json
│   │   │   └── preprocessing.pkl
│   │   │
│   │   ├── synthetic_ehr/
│   │   ├── maternal_health/
│   │   └── npha/
│   │
│   └── external/
│       └── clinical_guidelines/
│           ├── raw_docs/
│           ├── cleaned_text/
│           ├── chunks/
│           └── metadata/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_model_training_baselines.ipynb
│   ├── 03_shap_analysis.ipynb
│   ├── 04_dice_counterfactuals.ipynb
│   ├── 05_guardrail_retrieval.ipynb
│   ├── 06_branch_agent_demo.ipynb
│   └── 07_results_visualization.ipynb
│
├── src/
│   └── branch/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py
│       │   ├── preprocessors.py
│       │   ├── splitters.py
│       │   ├── feature_metadata.py
│       │   └── validation.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── train_xgboost.py
│       │   ├── train_baselines.py
│       │   ├── predict.py
│       │   ├── tune_optuna.py
│       │   └── registry.py
│       │
│       ├── explainability/
│       │   ├── __init__.py
│       │   ├── shap_explainer.py
│       │   ├── dice_counterfactual.py
│       │   ├── explanation_schema.py
│       │   └── static_baselines.py
│       │
│       ├── guardrails/
│       │   ├── __init__.py
│       │   ├── guideline_loader.py
│       │   ├── chunker.py
│       │   ├── embeddings.py
│       │   ├── vector_store.py
│       │   ├── retriever.py
│       │   ├── alignment_checker.py
│       │   └── anomaly_detector.py
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── react_agent.py
│       │   ├── prompts.py
│       │   ├── tools.py
│       │   ├── narrative_generator.py
│       │   └── safety_filters.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── predictive_metrics.py
│       │   ├── explanation_metrics.py
│       │   ├── guardrail_metrics.py
│       │   ├── latency.py
│       │   ├── clinician_rubric.py
│       │   └── aggregate_results.py
│       │
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── shap_plots.py
│       │   ├── counterfactual_plots.py
│       │   ├── latency_plots.py
│       │   └── tables.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── logging.py
│           ├── io.py
│           ├── seeds.py
│           └── constants.py
│
├── scripts/
│   ├── download_datasets.py
│   ├── preprocess_all.py
│   ├── train_all_models.py
│   ├── run_shap_all.py
│   ├── run_dice_all.py
│   ├── build_guideline_index.py
│   ├── run_branch_explanations.py
│   ├── evaluate_predictions.py
│   ├── evaluate_explanations.py
│   └── generate_paper_tables.py
│
├── experiments/
│   ├── gallstone/
│   │   ├── seed_42/
│   │   ├── seed_123/
│   │   └── seed_999/
│   │
│   ├── synthetic_ehr/
│   ├── maternal_health/
│   └── npha/
│
├── artifacts/
│   ├── models/
│   │   ├── gallstone/
│   │   │   ├── xgb_model.pkl
│   │   │   ├── rf_model.pkl
│   │   │   └── lr_model.pkl
│   │   ├── synthetic_ehr/
│   │   ├── maternal_health/
│   │   └── npha/
│   │
│   ├── explainers/
│   │   ├── shap/
│   │   └── dice/
│   │
│   ├── vector_store/
│   │   ├── faiss_index/
│   │   └── chroma_index/
│   │
│   ├── predictions/
│   │   ├── gallstone_predictions.csv
│   │   ├── synthetic_ehr_predictions.csv
│   │   ├── maternal_health_predictions.csv
│   │   └── npha_predictions.csv
│   │
│   ├── explanations/
│   │   ├── shap_json/
│   │   ├── dice_json/
│   │   ├── branch_traces/
│   │   └── narratives/
│   │
│   └── logs/
│       ├── training/
│       ├── inference/
│       └── errors/
│
├── results/
│   ├── metrics/
│   │   ├── predictive_metrics.csv
│   │   ├── explanation_quality.csv
│   │   ├── guardrail_metrics.csv
│   │   ├── latency_metrics.csv
│   │   └── clinician_scores.csv
│   │
│   ├── tables/
│   │   ├── table_classification_results.tex
│   │   ├── table_regression_results.tex
│   │   ├── table_eqs.tex
│   │   ├── table_guardrail_latency.tex
│   │   └── table_likert_scores.tex
│   │
│   ├── figures/
│   │   ├── architecture_diagram.png
│   │   ├── workflow_diagram.png
│   │   ├── shap_gallstone_beeswarm.png
│   │   ├── shap_maternal_summary.png
│   │   ├── counterfactual_distance_violin.png
│   │   ├── latency_breakdown.png
│   │   └── guardrail_confusion_matrix.png
│   │
│   └── qualitative/
│       ├── gallstone_case_42.md
│       ├── maternal_patient_17.md
│       ├── synthetic_ehr_anomaly.md
│       └── npha_regression_case.md
│
├── paper/
│   ├── main.tex
│   ├── sections/
│   │   ├── 01_introduction.tex
│   │   ├── 02_related_work.tex
│   │   ├── 03_datasets.tex
│   │   ├── 04_method.tex
│   │   ├── 05_experiments.tex
│   │   ├── 06_results.tex
│   │   ├── 07_qualitative.tex
│   │   ├── 08_discussion.tex
│   │   └── 09_conclusion.tex
│   │
│   ├── figures/
│   ├── tables/
│   ├── references.bib
│   └── branch_paper.pdf
│
└── tests/
    ├── test_data_loading.py
    ├── test_preprocessing.py
    ├── test_prediction.py
    ├── test_shap_output.py
    ├── test_dice_constraints.py
    ├── test_guideline_retrieval.py
    ├── test_alignment_checker.py
    └── test_narrative_schema.py
```

---

# 11. Important Configuration Files

## 11.1 `configs/default.yaml`

```yaml
project:
  name: BRANCH
  seed: 42
  output_dir: artifacts
  results_dir: results

experiment:
  seeds: [42, 123, 999]
  test_size: 0.2
  cv_folds: 5

model:
  primary: xgboost
  tune_with_optuna: true

explainability:
  shap_top_k: 5
  generate_counterfactuals_for_high_risk: true

llm:
  provider: local_or_api
  model_name: replace_with_available_model
  temperature: 0.0
  max_tokens: 800

guardrail:
  vector_store: faiss
  top_k_retrieval: 3
  anomaly_check: true

logging:
  level: INFO
  save_traces: true
```

---

## 11.2 `configs/models/xgboost.yaml`

```yaml
xgboost:
  objective_classification: binary:logistic
  objective_multiclass: multi:softprob
  objective_regression: reg:squarederror
  eval_metric_classification: logloss
  eval_metric_regression: rmse

optuna_search_space:
  max_depth: [3, 10]
  n_estimators: [100, 600]
  learning_rate: [0.01, 0.3]
  subsample: [0.6, 1.0]
  colsample_bytree: [0.5, 1.0]
```

---

## 11.3 `configs/branch/dice.yaml`

```yaml
dice:
  total_counterfactuals: 3
  desired_class: opposite
  proximity_weight: 0.5
  diversity_weight: 1.0
  permitted_range_file: feature_metadata.json
  immutable_features:
    - Age
    - Sex
  method: random
```

---

## 11.4 `configs/branch/guardrail.yaml`

```yaml
guardrail:
  retrieval:
    top_k: 3
    similarity_threshold: 0.35

  anomaly_detection:
    enabled: true
    require_guideline_support: true
    contradiction_policy: flag

  statuses:
    - no_anomaly
    - possible_anomaly
    - anomaly_detected
    - insufficient_guideline_evidence
    - retrieval_failed
```

---

# 12. Results Structure

Every experiment should save results in a reproducible, paper-ready format.

## 12.1 Predictive Results

Save as:

```text
results/metrics/predictive_metrics.csv
```

Recommended columns:

```text
dataset
seed
method
task_type
accuracy
macro_f1
precision
recall
auroc
mae
rmse
r2
train_time_sec
inference_time_sec
```

Example:

```csv
dataset,seed,method,task_type,accuracy,macro_f1,auroc,mae,rmse,r2
maternal_health,42,xgboost,multiclass,0.83,0.81,0.89,,,
npha,42,xgboost,regression,,,,1.12,1.64,0.42
```

---

## 12.2 SHAP Explanation Results

Save per patient:

```text
artifacts/explanations/shap_json/{dataset}/{patient_id}.json
```

Schema:

```json
{
  "dataset": "gallstone",
  "patient_id": 42,
  "model": "xgboost",
  "base_value": 0.41,
  "prediction": 0.85,
  "top_k": 5,
  "features": [
    {
      "rank": 1,
      "feature": "TotalCholesterol",
      "value": 245,
      "shap": 0.21,
      "direction": "increases_prediction"
    }
  ]
}
```

---

## 12.3 DiCE Results

Save per patient:

```text
artifacts/explanations/dice_json/{dataset}/{patient_id}.json
```

Schema:

```json
{
  "dataset": "maternal_health",
  "patient_id": 17,
  "original_prediction": "High Risk",
  "counterfactual_prediction": "Mid Risk",
  "distance": 0.18,
  "changes": [
    {
      "feature": "SystolicBP",
      "from": 150,
      "to": 135,
      "mutable": true
    }
  ],
  "validity": "valid"
}
```

---

## 12.4 Guardrail Results

Save per patient:

```text
artifacts/explanations/branch_traces/{dataset}/{patient_id}_guardrail.json
```

Schema:

```json
{
  "dataset": "synthetic_ehr",
  "patient_id": 309,
  "guardrail_status": "anomaly_detected",
  "retrieved_chunks": [
    {
      "source": "AHA guideline",
      "chunk_id": "aha_lipid_003",
      "relevance_score": 0.82,
      "summary": "High HDL is generally considered protective in cardiovascular risk assessment."
    }
  ],
  "alignment_checks": [
    {
      "feature": "HDLCholesterol",
      "model_direction": "increases_risk",
      "guideline_direction": "protective",
      "alignment": "discordant"
    }
  ]
}
```

---

## 12.5 Final Narrative Results

Save as Markdown or JSON.

```text
artifacts/explanations/narratives/{dataset}/{patient_id}.md
```

Example:

```md
# BRANCH Explanation: Patient 42

## Prediction
The model predicts high gallstone risk with 85% probability.

## Main Drivers
The largest upward contributors were total cholesterol, age, and BMI. Physical activity slightly reduced the predicted risk.

## Clinical Interpretation
Retrieved guideline evidence supports elevated cholesterol as a plausible contributor to gallstone-related risk.

## Counterfactual
The model would move toward a lower-risk prediction if total cholesterol decreased by approximately 40 mg/dL and BMI decreased by 5%, assuming all other features remain fixed.

## Guardrail Status
No anomaly detected.

## Caution
This is a model explanation, not a medical diagnosis. Clinician review is required.
```

---

## 12.6 Explanation Quality Results

Save as:

```text
results/metrics/explanation_quality.csv
```

Columns:

```text
dataset
patient_id
method
faithfulness
completeness
clinical_alignment
eqs
notes
```

Example:

```csv
dataset,patient_id,method,faithfulness,completeness,clinical_alignment,eqs
maternal_health,17,branch,0.92,1.00,1.00,0.968
maternal_health,17,shap_static,1.00,1.00,0.60,0.88
```

---

# 13. Paper Tables to Generate

## Table 1: Dataset Summary

Columns:

```text
Dataset
N
Features
Task
Source
Target
```

---

## Table 2: Classification Performance

Datasets:

- Gallstone
- Synthetic EHR
- Maternal Health

Metrics:

```text
AUROC
Macro-F1
Accuracy
```

Methods:

```text
LR
RF
XGB-Base
XGB+SHAP-Static
XGB+DiCE-Static
BRANCH
```

---

## Table 3: Regression Performance

Dataset:

- NPHA

Metrics:

```text
MAE
RMSE
R2
```

---

## Table 4: Explanation Quality Score

Metrics:

```text
F_faith
F_comp
F_align
EQS
```

Methods:

```text
XGB+SHAP-Static
XGB+DiCE-Static
BRANCH-small-LLM
BRANCH-large-LLM
```

---

## Table 5: Guardrail and Latency

Metrics:

```text
Guardrail Precision
Guardrail Recall
Guardrail F1
Mean Latency
```

---

## Table 6: Clinician Likert Ratings

Metrics:

```text
Clarity
Accuracy
Actionability
Trustworthiness
```

---

# 14. Paper Figures to Generate

## Figure 1: BRANCH Architecture

Show:

```text
Clinician Query
    ↓
LLM ReAct Agent
    ↓
XGBoost Predictor ↔ SHAP Explainer ↔ DiCE Counterfactual
    ↓
Clinical Guideline Retriever
    ↓
Clinical Alignment Checker
    ↓
Final Narrative + Guardrail Status
```

---

## Figure 2: ReAct Workflow Trace

Show steps:

```text
1. Query parsing
2. Prediction call
3. SHAP call
4. DiCE call
5. Guideline retrieval
6. Alignment check
7. Narrative synthesis
```

---

## Figure 3: Global SHAP Beeswarm

For Gallstone dataset.

---

## Figure 4: Maternal Health Multi-Class SHAP Summary

Show feature importance for:

```text
Low Risk
Mid Risk
High Risk
```

---

## Figure 5: Counterfactual Distance Distribution

Violin plot or box plot across datasets.

---

## Figure 6: Latency Breakdown

Show time consumed by:

```text
Prediction
SHAP
DiCE
Retrieval
LLM generation
```

---

# 15. Implementation Milestones

## Milestone 1: Dataset and Baseline Setup

Deliverables:

- Dataset loaders
- Preprocessing scripts
- Train/test splits
- Baseline models
- Predictive metrics CSV

Expected files:

```text
scripts/preprocess_all.py
scripts/train_all_models.py
results/metrics/predictive_metrics.csv
```

---

## Milestone 2: SHAP Explanation Pipeline

Deliverables:

- Local SHAP explanations
- Global SHAP plots
- Top-k feature JSON outputs

Expected files:

```text
src/branch/explainability/shap_explainer.py
scripts/run_shap_all.py
artifacts/explanations/shap_json/
results/figures/shap_gallstone_beeswarm.png
```

---

## Milestone 3: DiCE Counterfactual Pipeline

Deliverables:

- Feature mutability metadata
- Counterfactual generator
- Counterfactual distance metrics
- Counterfactual plots

Expected files:

```text
src/branch/explainability/dice_counterfactual.py
data/processed/{dataset}/feature_metadata.json
artifacts/explanations/dice_json/
results/figures/counterfactual_distance_violin.png
```

---

## Milestone 4: Clinical Guideline Guardrail

Deliverables:

- Guideline corpus
- Chunking pipeline
- Vector index
- Retrieval module
- Alignment checker
- Anomaly detector

Expected files:

```text
src/branch/guardrails/guideline_loader.py
src/branch/guardrails/retriever.py
src/branch/guardrails/alignment_checker.py
src/branch/guardrails/anomaly_detector.py
artifacts/vector_store/
```

---

## Milestone 5: LLM Agent and Narrative Generator

Deliverables:

- Tool wrapper functions
- Prompt templates
- Agent execution loop
- Narrative generator
- Saved explanation traces

Expected files:

```text
src/branch/agents/react_agent.py
src/branch/agents/prompts.py
src/branch/agents/tools.py
src/branch/agents/narrative_generator.py
artifacts/explanations/narratives/
```

---

## Milestone 6: Evaluation and Paper Results

Deliverables:

- EQS calculation
- Guardrail evaluation
- Latency evaluation
- Paper-ready tables
- Paper-ready figures

Expected files:

```text
src/branch/evaluation/explanation_metrics.py
src/branch/evaluation/guardrail_metrics.py
src/branch/evaluation/latency.py
scripts/generate_paper_tables.py
results/tables/
results/figures/
```

---

# 16. Suggested Implementation Order

Follow this order to avoid getting stuck:

```text
1. Start with one dataset only: Maternal Health Risk
2. Build preprocessing and XGBoost training
3. Add SHAP explanation
4. Add DiCE with feature constraints
5. Add simple guideline retrieval with 5-10 manually curated chunks
6. Add rule-based alignment checker
7. Add LLM narrative generation
8. Save end-to-end output for 10 patients
9. Expand to Gallstone
10. Expand to NPHA and Synthetic EHR
11. Run all baselines
12. Generate final tables and figures
```

Do not start by implementing the full multi-dataset agent. First build one clean end-to-end case.

---

# 17. Minimal MVP Version

If compute or time is limited, implement this MVP:

```text
Dataset: Maternal Health Risk
Predictor: XGBoost
Explanation: SHAP top-5
Counterfactual: DiCE for high-risk patients
Guideline corpus: small curated maternal risk guideline set
Guardrail: rule-based feature-direction checker
LLM: prompt-based narrative generator
Evaluation: 50 sampled patients
Metrics: accuracy, macro-F1, EQS, latency
```

This MVP is enough to demonstrate the main idea.

---

# 18. Risks and How to Fix Them

## Risk 1: Reviewer says this is just tool wrapping

Fix:

- Emphasize the clinical guardrail anomaly detection.
- Define a formal algorithm.
- Evaluate explanation quality systematically.
- Show examples where static SHAP fails to flag clinical contradictions.

---

## Risk 2: LLM hallucination

Fix:

- Use structured JSON inputs.
- Use temperature 0.
- Restrict the LLM to SHAP and retrieved guideline evidence.
- Add a post-generation factuality checker.
- Save all tool traces.

---

## Risk 3: Counterfactuals are clinically impossible

Fix:

- Define mutable and immutable features.
- Add permitted ranges.
- Reject invalid counterfactuals.
- Report feasibility status.

---

## Risk 4: Weak clinical evaluation

Fix:

- Use a clear clinician rubric.
- Report inter-rater agreement if possible.
- Add automatic clinical alignment checks.
- Avoid claiming clinical deployment readiness.

---

## Risk 5: Dataset size is small

Fix:

- Use multiple datasets.
- Report results over multiple seeds.
- Avoid overclaiming generalization.
- Position BRANCH as an explanation framework, not a new clinical predictor.

---

# 19. Recommended README Summary

Use this in the repository README:

```md
# BRANCH: Boosted Reasoning and Agentic Narratives for Clinical Healthcare

BRANCH is an explainable clinical machine learning framework that wraps XGBoost predictors with SHAP attribution, DiCE counterfactual recourse, and retrieval-augmented clinical guardrails. An LLM-based ReAct agent orchestrates these tools to generate faithful, complete, and clinically grounded natural-language explanations for tabular healthcare predictions.

The system is evaluated across multiple public medical tabular datasets, including binary classification, multi-class risk prediction, synthetic EHR classification, and regression. BRANCH preserves XGBoost predictive performance while improving explanation quality over static SHAP and DiCE baselines.
```

---

# 20. Final Expected Contribution Statement

The final paper should state the contribution as:

> We propose BRANCH, a modular agentic explainability framework for structured clinical machine learning. BRANCH combines XGBoost prediction, SHAP attribution, DiCE counterfactual recourse, and vector-retrieved clinical guideline guardrails within an LLM-orchestrated ReAct loop. Across four public healthcare datasets, BRANCH preserves predictive performance while improving explanation faithfulness, completeness, clinical alignment, and actionability compared with static explanation baselines.

---

# 21. Final Checklist Before Writing Results

Before filling the final paper tables, make sure the following exist:

```text
[ ] Clean train/test splits for all datasets
[ ] XGBoost trained for all datasets
[ ] LR/RF/Ridge baselines trained
[ ] Predictive metrics saved over multiple seeds
[ ] SHAP local explanations saved as JSON
[ ] Global SHAP plots generated
[ ] DiCE counterfactuals generated with feature constraints
[ ] Guideline corpus built and indexed
[ ] Retrieval results saved
[ ] Alignment checker implemented
[ ] Guardrail anomalies logged
[ ] LLM narratives generated
[ ] Explanation quality metrics computed
[ ] Latency measured
[ ] Qualitative examples selected
[ ] Paper tables exported to LaTeX
[ ] Paper figures exported as PNG/PDF
```

---

# 22. Recommended Immediate Next Step

Build the end-to-end version for **Maternal Health Risk** first.

Why this dataset first?

- Small and easy to process.
- Features are clinically interpretable.
- Multi-class risk labels are useful for demonstration.
- SHAP explanations will be easy to understand.
- Counterfactuals such as blood pressure and glucose changes are intuitive.
- Guardrail checking is easier because maternal blood pressure and glucose risk directions are well defined.

Once Maternal Health Risk works end to end, replicate the same pipeline for the other datasets.