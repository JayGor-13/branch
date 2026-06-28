from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from branch.utils.dependencies import require


CLASSIFICATION_DATASETS = [
    ("gallstone", "Gallstone"),
    ("maternal_health", "Maternal Health"),
    ("npha", "NPHA"),
]
CLASSIFICATION_METHODS = [
    ("LR", "LR"),
    ("RF", "RF"),
    ("XGB-Base", "XGBoost"),
]
REGRESSION_METHODS = [
    ("Ridge", "LR (Ridge)"),
    ("RF", "RF"),
    ("XGB-Base", "XGBoost"),
]
EXPLANATION_VARIANTS = [
    "BRANCH-Gemma4-26B",
    "BRANCH-Gemma4-31B",
]
LATENCY_VARIANTS = [
    "BRANCH-Gemma4-26B",
    "BRANCH-Gemma4-31B",
]
EXPLANATION_DATASETS = {"gallstone", "maternal_health", "npha"}
TABLE_DATASET_LABELS = {
    "gallstone": "Gallstone Disease",
    "maternal_health": "Maternal Health",
    "npha": "NPHA",
    "load_diabetes": "load_diabetes",
}
LATENCY_DETAIL_COLUMNS = [
    "method",
    "dataset",
    "patient_id",
    "llm_provider",
    "llm_model",
    "llm_used",
    "llm_fallback_used",
    "retrieved_chunks",
    "xgboost_sec",
    "shap_sec",
    "dice_sec",
    "rag_retrieval_sec",
    "guardrail_sec",
    "llm_api_sec",
    "llm_summary_sec",
    "component_sum_sec",
    "total_inference_sec",
]
RAGAS_DETAIL_COLUMNS = [
    "method",
    "dataset",
    "patient_id",
    "faithfulness",
    "answer_relevancy",
    "clinical_alignment",
    "eqs",
    "alignment_mode",
    "rag_source",
    "rag_retrieved_chunks",
    "rag_backend",
    "embedding_provider",
    "embedding_model",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready BRANCH result tables."
    )
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--trace-root", default="artifacts/explanations/branch_traces")
    parser.add_argument(
        "--variant-artifact-root",
        default="artifacts/gemini_variants",
        help=(
            "Root containing variant-specific traces, for example "
            "artifacts/gemini_variants/gemma4_26b/explanations/branch_traces."
        ),
    )
    parser.add_argument(
        "--guardrail-benchmark-path",
        default="results/metrics/guardrail_seeded_benchmark.csv",
    )
    parser.add_argument("--output-dir", default="results/tables")
    parser.add_argument(
        "--table-iv-placeholder",
        action="store_true",
        help="Render Table IV with -- placeholders even if explanation metrics exist.",
    )
    parser.add_argument(
        "--table-iv-sample-size",
        type=int,
        default=50,
        help="Sample count shown in the Table IV caption.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictive = _read_csv(Path(args.metrics_dir) / "predictive_metrics.csv")
    explanation = _read_explanation_quality_files(Path(args.metrics_dir))

    table_ii = build_classification_table(predictive)
    table_iii = build_regression_table(predictive)
    table_iv = build_explanation_quality_table(
        explanation, placeholder=args.table_iv_placeholder
    )
    ragas_detail = build_ragas_detail_table(explanation)
    ragas_by_dataset = build_ragas_by_dataset_table(ragas_detail)
    latency_detail = build_latency_detail_table(Path(args.variant_artifact_root))
    table_v = build_latency_breakdown_table(latency_detail)
    table_v_by_dataset = build_latency_breakdown_by_dataset(latency_detail)

    _write_table_bundle(
        table_ii,
        output_dir / "table_ii_classification_results",
        latex_table_ii(table_ii),
    )
    _write_table_bundle(
        table_iii,
        output_dir / "table_iii_regression_results",
        latex_table_iii(table_iii),
    )
    _write_table_bundle(
        table_iv,
        output_dir / "table_iv_explanation_quality",
        latex_table_iv(table_iv, sample_size=args.table_iv_sample_size),
    )
    _write_table_bundle(
        table_v,
        output_dir / "table_v_guardrail_latency",
        latex_table_v(table_v),
    )
    _write_csv(ragas_detail, output_dir / "table_iv_ragas_per_patient.csv")
    _write_csv(ragas_by_dataset, output_dir / "table_iv_ragas_by_dataset.csv")
    _write_csv(latency_detail, output_dir / "table_v_latency_per_patient.csv")
    _write_csv(table_v_by_dataset, output_dir / "table_v_latency_by_dataset.csv")
    print(f"Saved paper tables to {output_dir}")


def build_classification_table(df):
    rows = []
    for method_key, method_label in CLASSIFICATION_METHODS:
        row = {"Method": method_label}
        for dataset, dataset_label in CLASSIFICATION_DATASETS:
            subset = _metric_rows(df, dataset, method_key)
            row[f"{dataset_label} AUROC"] = _format_metric(subset, "auroc")
            row[f"{dataset_label} F1"] = _format_metric(subset, "macro_f1")
            row[f"{dataset_label} Acc."] = _format_metric(
                subset, "accuracy", percent=True
            )
        rows.append(row)
    return _df(rows)


def build_regression_table(df):
    rows = []
    for method_key, method_label in REGRESSION_METHODS:
        subset = _metric_rows(df, "load_diabetes", method_key)
        rows.append(
            {
                "Method": method_label,
                "MAE": _format_metric(subset, "mae"),
                "RMSE": _format_metric(subset, "rmse"),
                "R2": _format_metric(subset, "r2"),
            }
        )
    return _df(rows)


def build_explanation_quality_table(df, placeholder: bool = False):
    rows = []
    for method in EXPLANATION_VARIANTS:
        subset = _explanation_variant_rows(df, method) if not placeholder else df.iloc[0:0]
        answer_column = "answer_relevancy" if "answer_relevancy" in subset else "completeness"
        rows.append(
            {
                "Method": method,
                "Faithfulness": _format_metric(subset, "faithfulness"),
                "Answer Relevancy": _format_metric(subset, answer_column),
            }
        )
    return _df(rows)


def build_ragas_detail_table(df):
    if df.empty or "method" not in df:
        return _df_with_columns(RAGAS_DETAIL_COLUMNS)

    rows = []
    for _, row in df.iterrows():
        method = str(row.get("method", ""))
        dataset = str(row.get("dataset", ""))
        if method not in EXPLANATION_VARIANTS or dataset not in EXPLANATION_DATASETS:
            continue
        if str(row.get("quality_mode", "local")) != "ragas":
            continue
        answer_relevancy = (
            row.get("answer_relevancy")
            if "answer_relevancy" in df
            else row.get("completeness")
        )
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "patient_id": row.get("patient_id"),
                "faithfulness": row.get("faithfulness"),
                "answer_relevancy": answer_relevancy,
                "clinical_alignment": row.get("clinical_alignment"),
                "eqs": row.get("eqs"),
                "alignment_mode": row.get("alignment_mode"),
                "rag_source": row.get("rag_source"),
                "rag_retrieved_chunks": row.get("rag_retrieved_chunks"),
                "rag_backend": row.get("rag_backend"),
                "embedding_provider": row.get("embedding_provider"),
                "embedding_model": row.get("embedding_model"),
                "notes": row.get("notes"),
            }
        )
    if not rows:
        return _df_with_columns(RAGAS_DETAIL_COLUMNS)
    return _df(rows)


def build_ragas_by_dataset_table(detail_df):
    rows = []
    for method in EXPLANATION_VARIANTS:
        method_rows = _ragas_rows(detail_df, method)
        rows.append(_ragas_summary_row(method, "Average", method_rows))
        for dataset in sorted(EXPLANATION_DATASETS):
            subset = _ragas_rows(detail_df, method, dataset=dataset)
            rows.append(
                _ragas_summary_row(
                    method,
                    TABLE_DATASET_LABELS.get(dataset, dataset),
                    subset,
                )
            )
    return _df(rows)


def build_latency_detail_table(variant_artifact_root: Path):
    rows = []
    for method in LATENCY_VARIANTS:
        trace_root = (
            variant_artifact_root
            / _variant_slug(method)
            / "explanations"
            / "branch_traces"
        )
        for dataset in sorted(EXPLANATION_DATASETS):
            for trace in _load_traces(trace_root / dataset):
                backend = trace.get("narrative_backend", {})
                retrieval = trace.get("retrieval_backend", {})
                latencies = trace.get("latency_sec", {})
                xgboost = _latency_value(latencies, "prediction")
                shap = _latency_value(latencies, "shap")
                dice = _latency_value(latencies, "dice")
                rag = _latency_value(latencies, "guideline_retrieval")
                guardrail = _latency_value(latencies, "clinical_alignment")
                llm_summary = _latency_value(latencies, "narrative_generation")
                llm_api = _optional_float(backend.get("llm_api_latency_sec"))
                component_sum = xgboost + shap + dice + rag + guardrail + llm_summary
                rows.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        "patient_id": trace.get("patient_id"),
                        "llm_provider": backend.get("provider"),
                        "llm_model": backend.get("model_name"),
                        "llm_used": backend.get("used_llm"),
                        "llm_fallback_used": backend.get("fallback_used"),
                        "retrieved_chunks": retrieval.get("retrieved_chunks"),
                        "xgboost_sec": xgboost,
                        "shap_sec": shap,
                        "dice_sec": dice,
                        "rag_retrieval_sec": rag,
                        "guardrail_sec": guardrail,
                        "llm_api_sec": llm_api,
                        "llm_summary_sec": llm_summary,
                        "component_sum_sec": component_sum,
                        "total_inference_sec": float(
                            trace.get("total_latency_sec", component_sum)
                        ),
                    }
                )
    if not rows:
        return _df_with_columns(LATENCY_DETAIL_COLUMNS)
    return _df(rows)


def build_latency_breakdown_table(detail_df):
    rows = []
    for method in LATENCY_VARIANTS:
        subset = _latency_rows(detail_df, method)
        rows.append(_latency_summary_row(method, subset))
    return _df(rows)


def build_latency_breakdown_by_dataset(detail_df):
    rows = []
    for method in LATENCY_VARIANTS:
        for dataset in sorted(EXPLANATION_DATASETS):
            subset = _latency_rows(detail_df, method, dataset=dataset)
            summary = _latency_summary_row(method, subset)
            row = {
                "Method": summary["Method"],
                "Dataset": TABLE_DATASET_LABELS.get(dataset, dataset),
                "XGBoost": summary["XGBoost"],
                "SHAP": summary["SHAP"],
                "DiCE": summary["DiCE"],
                "RAG": summary["RAG"],
                "Guardrail": summary["Guardrail"],
                "LLM Summary": summary["LLM Summary"],
                "Total": summary["Total"],
            }
            rows.append(row)
    return _df(rows)


def latex_table_ii(df) -> str:
    rows = []
    for _, row in df.iterrows():
        values = [row["Method"]]
        for _, dataset_label in CLASSIFICATION_DATASETS:
            values.extend(
                [
                    row[f"{dataset_label} AUROC"],
                    row[f"{dataset_label} F1"],
                    row[f"{dataset_label} Acc."],
                ]
            )
        rows.append(" & ".join(_latex_escape(value) for value in values) + r" \\")
    body = "\n".join(rows)
    return rf"""\begin{{table*}}[ht]
\centering
\caption{{\textbf{{Predictor Selection (Classification).}} AUROC, macro-F1, and accuracy (\%) for candidate models on the three classification datasets. All values are mean $\pm$ std over available seeds.}}
\label{{tab:clf_results}}
\renewcommand{{\arraystretch}}{{1.2}}
\begin{{tabular}}{{l ccc ccc ccc}}
\toprule
& \multicolumn{{3}}{{c}}{{\textbf{{Gallstone}}}} & \multicolumn{{3}}{{c}}{{\textbf{{Maternal Health}}}} & \multicolumn{{3}}{{c}}{{\textbf{{NPHA}}}} \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}\cmidrule(lr){{8-10}}
\textbf{{Method}} & AUROC & F1 & Acc. & AUROC & F1 & Acc. & AUROC & F1 & Acc. \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def latex_table_iii(df) -> str:
    body = "\n".join(
        " & ".join(
            _latex_escape(row[column]) for column in ["Method", "MAE", "RMSE", "R2"]
        )
        + r" \\"
        for _, row in df.iterrows()
    )
    return rf"""\begin{{table}}[ht]
\centering
\caption{{\textbf{{Predictor Selection (Regression).}} MAE, RMSE, and $R^2$ for predicting disease progression one year after baseline on \texttt{{load\_diabetes}}. All values are mean $\pm$ std over available seeds.}}
\label{{tab:reg_results}}
\renewcommand{{\arraystretch}}{{1.2}}
\begin{{tabular}}{{lccc}}
\toprule
\textbf{{Method}} & \textbf{{MAE}} $\downarrow$ & \textbf{{RMSE}} $\downarrow$ & \textbf{{$R^2$}} $\uparrow$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def latex_table_iv(df, sample_size: int = 50) -> str:
    rows = []
    for _, row in df.iterrows():
        method = str(row["Method"])
        values = [
            _latex_method_label(method),
            _latex_escape(row["Faithfulness"]),
            _latex_escape(row["Answer Relevancy"]),
        ]
        if method == "BRANCH-Gemma4-31B":
            values[1] = rf"\textbf{{{values[1]}}}"
            values[2] = rf"\textbf{{{values[2]}}}"
        rows.append(" & ".join(values) + r" \\")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[ht]
\centering
\caption{{\textbf{{Explanation Quality Score (EQS) via RAGAS.}} Faithfulness and Answer Relevancy reported for each BRANCH variant, computed over {sample_size} sampled test patients per dataset and averaged across classification datasets. Higher is better (max = 1.0). Static baselines are excluded as they perform no guideline retrieval. \textit{{[Fill in after experiments.]}}}}
\label{{tab:eqs}}
\renewcommand{{\arraystretch}}{{1.2}}
\begin{{tabular}}{{lcc}}
\toprule
\textbf{{Method}} & \textbf{{Faithfulness}} & \textbf{{Answer Relevancy}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def latex_table_v(df) -> str:
    body = "\n".join(
        " & ".join(
            _latex_escape(row[column])
            for column in [
                "Method",
                "XGBoost",
                "SHAP",
                "DiCE",
                "RAG",
                "Guardrail",
                "LLM Summary",
                "Total",
            ]
        )
        + r" \\"
        for _, row in df.iterrows()
    )
    return rf"""\begin{{table*}}[ht]
\centering
\caption{{\textbf{{Inference Latency Breakdown.}} Mean seconds $\pm$ std over generated BRANCH traces. XGBoost is model prediction time, SHAP and DiCE are explanation tool times, RAG is top-3 guideline retrieval, Guardrail is clinical-alignment checking, and LLM Summary is the Gemini API generation call grounded in the retrieved chunks. Total is end-to-end inference time.}}
\label{{tab:latency_breakdown}}
\renewcommand{{\arraystretch}}{{1.2}}
\begin{{tabular}}{{lccccccc}}
\toprule
\textbf{{Method}} & \textbf{{XGBoost}} & \textbf{{SHAP}} & \textbf{{DiCE}} & \textbf{{RAG}} & \textbf{{Guardrail}} & \textbf{{LLM Summary}} & \textbf{{Total}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def _read_csv(path: Path):
    pd = require("pandas")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_explanation_quality_files(metrics_dir: Path):
    pd = require("pandas")
    frames = [
        pd.read_csv(path)
        for path in sorted(metrics_dir.glob("explanation_quality*.csv"))
        if path.is_file()
    ]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "quality_mode" not in df:
        df["quality_mode"] = "local"
    else:
        df["quality_mode"] = df["quality_mode"].fillna("local")
    return df


def _explanation_variant_rows(df, method: str):
    if df.empty or "method" not in df:
        return df
    subset = df[df["method"].fillna("") == method]
    if "quality_mode" in subset:
        subset = subset[subset["quality_mode"] == "ragas"]
    if "dataset" in subset:
        subset = subset[subset["dataset"].isin(EXPLANATION_DATASETS)]
    return subset


def _metric_rows(df, dataset: str, method: str):
    if df.empty:
        return df
    return df[(df["dataset"] == dataset) & (df["method"] == method)]


def _format_metric(rows, column: str, percent: bool = False) -> str:
    if rows.empty or column not in rows:
        return "--"
    values = [
        float(value)
        for value in rows[column].dropna().tolist()
        if str(value).strip() != ""
    ]
    if percent:
        values = [value * 100.0 for value in values]
    return _format_values(values)


def _format_values(values: list[float]) -> str:
    if not values:
        return "--"
    avg = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    return f"{avg:.3f} ± {sd:.3f}"


def _ragas_rows(detail_df, method: str, dataset: str | None = None):
    if detail_df.empty:
        return detail_df
    subset = detail_df[detail_df["method"] == method]
    if dataset is not None:
        subset = subset[subset["dataset"] == dataset]
    return subset


def _ragas_summary_row(method: str, dataset_label: str, rows) -> dict[str, str]:
    return {
        "Method": _display_method(method),
        "Dataset": dataset_label,
        "Faithfulness": _format_metric(rows, "faithfulness"),
        "Answer Relevancy": _format_metric(rows, "answer_relevancy"),
        "Clinical Alignment": _format_metric(rows, "clinical_alignment"),
        "EQS": _format_metric(rows, "eqs"),
        "Retrieved Chunks": _format_metric(rows, "rag_retrieved_chunks"),
    }


def _latency_value(latencies: dict[str, Any], key: str) -> float:
    value = latencies.get(key, 0.0)
    if value is None or str(value).strip() == "":
        return 0.0
    return float(value)


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _latency_rows(detail_df, method: str, dataset: str | None = None):
    if detail_df.empty:
        return detail_df
    subset = detail_df[detail_df["method"] == method]
    if dataset is not None:
        subset = subset[subset["dataset"] == dataset]
    return subset


def _latency_summary_row(method: str, rows) -> dict[str, str]:
    return {
        "Method": _display_method(method),
        "XGBoost": _format_metric(rows, "xgboost_sec"),
        "SHAP": _format_metric(rows, "shap_sec"),
        "DiCE": _format_metric(rows, "dice_sec"),
        "RAG": _format_metric(rows, "rag_retrieval_sec"),
        "Guardrail": _format_metric(rows, "guardrail_sec"),
        "LLM Summary": _format_metric(rows, "llm_summary_sec"),
        "Total": _format_metric(rows, "total_inference_sec"),
    }


def _variant_slug(method_label: str) -> str:
    return method_label.lower().replace("branch-", "").replace("-", "_")


def _load_traces(trace_dir: Path) -> list[dict[str, Any]]:
    traces = []
    if not trace_dir.exists():
        return traces
    for path in sorted(trace_dir.glob("*_trace.json")):
        try:
            traces.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return traces


def _guardrail_metrics(df, dataset: str | None) -> dict[str, str]:
    if df.empty or "true_anomaly" not in df or "predicted_anomaly" not in df:
        return {"precision": "--", "recall": "--", "f1": "--"}
    subset = df if dataset is None else df[df["dataset"] == dataset]
    if subset.empty:
        return {"precision": "--", "recall": "--", "f1": "--"}
    true_values = [int(value) for value in subset["true_anomaly"].tolist()]
    pred_values = [int(value) for value in subset["predicted_anomaly"].tolist()]
    tp = sum(1 for true, pred in zip(true_values, pred_values) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(true_values, pred_values) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(true_values, pred_values) if true == 1 and pred == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": f"{precision:.3f}",
        "recall": f"{recall:.3f}",
        "f1": f"{f1:.3f}",
    }


def _write_table_bundle(df, stem: Path, latex: str) -> None:
    csv_path = stem.with_suffix(".csv")
    tex_path = stem.with_suffix(".tex")
    try:
        df.to_csv(csv_path, index=False)
        tex_path.write_text(latex, encoding="utf-8")
    except PermissionError:
        fallback_stem = stem.with_name(stem.name + "_new")
        fallback_csv = fallback_stem.with_suffix(".csv")
        fallback_tex = fallback_stem.with_suffix(".tex")
        df.to_csv(fallback_csv, index=False)
        fallback_tex.write_text(latex, encoding="utf-8")
        print(
            f"Warning: {csv_path} or {tex_path} is locked. "
            f"Wrote {fallback_csv} and {fallback_tex} instead."
        )


def _write_csv(df, path: Path) -> None:
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        fallback = path.with_name(path.stem + "_new" + path.suffix)
        df.to_csv(fallback, index=False)
        print(f"Warning: {path} is locked. Wrote {fallback} instead.")


def _display_method(method: str) -> str:
    if method.lower() == "branch":
        return "BRANCH"
    return method


def _latex_method_label(method: str) -> str:
    escaped = _latex_escape(method)
    if method == "BRANCH-Gemma4-31B":
        return rf"\textbf{{{escaped}}}"
    return escaped


def _df(rows: list[dict[str, Any]]):
    pd = require("pandas")
    return pd.DataFrame(rows)


def _df_with_columns(columns: list[str]):
    pd = require("pandas")
    return pd.DataFrame(columns=columns)


def _latex_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("±", r"$\pm$")
    )


if __name__ == "__main__":
    main()
