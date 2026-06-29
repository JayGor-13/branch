"""ReAct-style BRANCH orchestration without exposing hidden reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from branch.agents.llm_client import build_llm_client, build_llm_config
from branch.agents.tools import BranchTools
from branch.data.dataset_specs import default_model_path, default_processed_dir, normalize_dataset_name
from branch.data.feature_metadata import infer_feature_metadata
from branch.data.loaders import feature_columns_for_dataframe, load_processed_split
from branch.models.registry import load_model_bundle
from branch.rag.embeddings import build_embedding_client, build_embedding_config
from branch.utils.constants import PATIENT_ID
from branch.utils.io import ensure_dir, read_json, write_json, write_text


@dataclass
class BranchAgent:
    tools: BranchTools
    test_df: Any
    artifact_root: Path = Path("artifacts")

    @classmethod
    def from_artifacts(
        cls,
        dataset: str = "maternal_health",
        processed_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        artifact_root: str | Path = "artifacts",
        shap_top_k: int = 5,
        llm_provider: str = "template",
        llm_model: str = "deterministic_template_generator",
        llm_base_url: str | None = None,
        llm_api_key_env: str | None = None,
        llm_timeout_sec: int = 120,
        llm_max_tokens: int = 1600,
        llm_max_retries: int = 3,
        llm_retry_backoff_sec: float = 2.0,
        llm_fallback_to_template: bool = True,
        embedding_provider: str = "local",
        embedding_model: str | None = None,
        embedding_base_url: str | None = None,
        embedding_api_key_env: str | None = None,
        embedding_timeout_sec: int = 120,
        embedding_dimensions: int = 768,
        vector_index_path: str | Path = "artifacts/vector_store/clinical_guidelines",
        retrieval_top_k: int = 3,
        retrieval_similarity_threshold: float = 0.0,
    ) -> "BranchAgent":
        dataset = normalize_dataset_name(dataset)
        processed_dir = Path(processed_dir or default_processed_dir(dataset))
        model_path = Path(model_path or default_model_path(dataset))
        train_df, test_df = load_processed_split(dataset, processed_dir)
        bundle = load_model_bundle(model_path)
        metadata_path = processed_dir / "feature_metadata.json"
        metadata = (
            read_json(metadata_path)
            if metadata_path.exists()
            else infer_feature_metadata(
                dataset, train_df, feature_columns_for_dataframe(dataset, train_df)
            )
        )
        llm_config = build_llm_config(
            provider=llm_provider,
            model_name=llm_model,
            base_url=llm_base_url,
            api_key_env=llm_api_key_env,
            timeout_sec=llm_timeout_sec,
            max_tokens=llm_max_tokens,
            max_retries=llm_max_retries,
            retry_backoff_sec=llm_retry_backoff_sec,
            fallback_to_template=llm_fallback_to_template,
        )
        embedding_config = build_embedding_config(
            provider=embedding_provider,
            model_name=embedding_model,
            base_url=embedding_base_url,
            api_key_env=embedding_api_key_env,
            timeout_sec=embedding_timeout_sec,
            dimensions=embedding_dimensions,
        )
        tools = BranchTools(
            model=bundle.model,
            train_df=train_df,
            feature_names=bundle.feature_names,
            label_order=bundle.label_order,
            feature_metadata=metadata,
            dataset=dataset,
            task_type=bundle.task_type,
            shap_top_k=shap_top_k,
            llm_client=build_llm_client(llm_config),
            llm_fallback_to_template=llm_config.fallback_to_template,
            embedding_client=build_embedding_client(embedding_config),
            vector_index_path=Path(vector_index_path),
            retrieval_top_k=retrieval_top_k,
            retrieval_similarity_threshold=retrieval_similarity_threshold,
        )
        return cls(tools=tools, test_df=test_df, artifact_root=Path(artifact_root))

    def explain_patient(self, patient_id: int, query: str) -> dict[str, Any]:
        row = self._patient_by_id(patient_id)
        patient = row.to_dict()
        latencies: dict[str, float] = {}
        tools_called: list[str] = []

        prediction, elapsed = self._timed(lambda: self.tools.predict_xgboost(patient))
        latencies["prediction"] = elapsed
        tools_called.append("predict_xgboost")

        shap_result, elapsed = self._timed(lambda: self.tools.explain_shap(patient))
        latencies["shap"] = elapsed
        tools_called.append("explain_shap")
        self._save_shap(shap_result, patient_id)

        dice_result = None
        if should_generate_counterfactual(query, prediction):
            dice_result, elapsed = self._timed(lambda: self._safe_dice(patient, prediction))
            latencies["dice"] = elapsed
            tools_called.append("generate_counterfactual_dice")
            self._save_dice(dice_result, patient_id)

        guideline_context, elapsed = self._timed(
            lambda: self.tools.retrieve_guidelines(prediction, shap_result)
        )
        latencies["guideline_retrieval"] = elapsed
        tools_called.append("retrieve_guidelines")

        guardrail_result, elapsed = self._timed(
            lambda: self.tools.check_clinical_alignment(shap_result, guideline_context)
        )
        latencies["clinical_alignment"] = elapsed
        tools_called.append("check_clinical_alignment")

        narrative_result, elapsed = self._timed(
            lambda: self.tools.generate_narrative(
                query,
                prediction,
                shap_result,
                dice_result,
                guideline_context,
                guardrail_result,
            )
        )
        narrative = narrative_result.text
        latencies["narrative_generation"] = elapsed
        tools_called.append("generate_narrative")

        trace = {
            "query_id": f"q_{patient_id:06d}",
            "dataset": self.tools.dataset,
            "patient_id": int(patient_id),
            "tools_called": tools_called,
            "prediction": prediction,
            "shap_result_path": str(self._shap_path(patient_id)),
            "dice_result_path": str(self._dice_path(patient_id)) if dice_result else None,
            "guideline_context": guideline_context,
            "guardrail_result": guardrail_result,
            "guardrail_status": guardrail_result.get("guardrail_status"),
            "narrative_backend": {
                "provider": getattr(self.tools.llm_client, "provider", "template"),
                "model_name": getattr(
                    self.tools.llm_client,
                    "model_name",
                    "deterministic_template_generator",
                ),
                "base_url": getattr(
                    getattr(self.tools.llm_client, "config", None),
                    "base_url",
                    None,
                ),
                "api_key_env": getattr(
                    getattr(self.tools.llm_client, "config", None),
                    "api_key_env",
                    None,
                ),
                "fallback_to_template": self.tools.llm_fallback_to_template,
                "used_llm": narrative_result.used_llm,
                "fallback_used": narrative_result.fallback_used,
                "fallback_reason": narrative_result.fallback_reason,
                "llm_api_latency_sec": (
                    round(narrative_result.llm_latency_sec, 4)
                    if narrative_result.llm_latency_sec is not None
                    else None
                ),
            },
            "retrieval_backend": {
                "backend": guideline_context.get("retrieval_backend"),
                "vector_index_path": guideline_context.get("vector_index_path"),
                "embedding_provider": guideline_context.get("embedding_provider"),
                "embedding_model": guideline_context.get("embedding_model"),
                "retrieved_chunks": len(guideline_context.get("retrieved_chunks", [])),
            },
            "latency_sec": {key: round(value, 4) for key, value in latencies.items()},
            "total_latency_sec": round(sum(latencies.values()), 4),
            "final_output_id": f"narrative_{patient_id:06d}",
            "narrative_path": str(self._narrative_path(patient_id)),
        }

        self._save_trace(trace, patient_id)
        write_text(narrative, self._narrative_path(patient_id))
        return {"trace": trace, "narrative": narrative}

    def _patient_by_id(self, patient_id: int):
        matches = self.test_df[self.test_df[PATIENT_ID] == patient_id]
        if matches.empty:
            raise KeyError(f"Patient id {patient_id} was not found in the test split.")
        return matches.iloc[0]

    def _safe_dice(self, patient: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.tools.generate_counterfactual_dice(patient)
        except ImportError as exc:
            return {
                "dataset": self.tools.dataset,
                "patient_id": int(patient[PATIENT_ID]),
                "original_prediction": prediction["predicted_class"],
                "counterfactual_prediction": None,
                "distance": None,
                "changes": [],
                "validity": "dependency_missing",
                "feasibility_status": "dependency_missing",
                "notes": str(exc),
            }

    @staticmethod
    def _timed(fn):
        start = perf_counter()
        result = fn()
        return result, perf_counter() - start

    def _shap_path(self, patient_id: int) -> Path:
        return (
            self.artifact_root
            / "explanations"
            / "shap_json"
            / self.tools.dataset
            / f"{patient_id}.json"
        )

    def _dice_path(self, patient_id: int) -> Path:
        return (
            self.artifact_root
            / "explanations"
            / "dice_json"
            / self.tools.dataset
            / f"{patient_id}.json"
        )

    def _trace_path(self, patient_id: int) -> Path:
        return (
            self.artifact_root
            / "explanations"
            / "branch_traces"
            / self.tools.dataset
            / f"{patient_id}_trace.json"
        )

    def _narrative_path(self, patient_id: int) -> Path:
        return (
            self.artifact_root
            / "explanations"
            / "narratives"
            / self.tools.dataset
            / f"{patient_id}.md"
        )

    def _save_shap(self, result: dict[str, Any], patient_id: int) -> None:
        write_json(result, self._shap_path(patient_id))

    def _save_dice(self, result: dict[str, Any], patient_id: int) -> None:
        write_json(result, self._dice_path(patient_id))

    def _save_trace(self, result: dict[str, Any], patient_id: int) -> None:
        ensure_dir(self._trace_path(patient_id).parent)
        write_json(result, self._trace_path(patient_id))


def should_generate_counterfactual(query: str, prediction: dict[str, Any]) -> bool:
    if prediction.get("task_type") == "regression":
        return False
    query_l = query.lower()
    action_terms = ["counterfactual", "action", "modifiable", "what if", "lower risk"]
    label = str(prediction.get("predicted_class", "")).lower()
    high_or_positive = any(term in label for term in ["high", "present", "positive"])
    return high_or_positive or any(term in query_l for term in action_terms)
