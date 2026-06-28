"""Tool wrappers used by the BRANCH ReAct-style orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from branch.agents.narrative_generator import generate_narrative_result
from branch.explainability.dice_counterfactual import generate_dice_counterfactual
from branch.explainability.shap_explainer import explain_patient_shap
from branch.guardrails.alignment_checker import check_clinical_alignment
from branch.guardrails.retriever import retrieve_guidelines
from branch.models.predict import predict_patient


@dataclass
class BranchTools:
    model: Any
    train_df: Any
    feature_names: list[str]
    label_order: list[str]
    feature_metadata: dict[str, Any]
    dataset: str = "maternal_health"
    task_type: str = "multiclass_classification"
    shap_top_k: int = 5
    llm_client: Any | None = None
    llm_fallback_to_template: bool = True
    embedding_client: Any | None = None
    vector_index_path: Path = Path("artifacts/vector_store/maternal_health_guidelines")
    retrieval_top_k: int = 3
    retrieval_similarity_threshold: float = 0.0

    def predict_xgboost(self, patient: dict[str, Any]) -> dict[str, Any]:
        return predict_patient(
            self.model,
            patient,
            self.feature_names,
            self.label_order,
            self.dataset,
            task_type=self.task_type,
        )

    def explain_shap(self, patient: dict[str, Any]) -> dict[str, Any]:
        return explain_patient_shap(
            self.model,
            patient,
            self.feature_names,
            self.label_order,
            self.dataset,
            task_type=self.task_type,
            top_k=self.shap_top_k,
        )

    def generate_counterfactual_dice(self, patient: dict[str, Any]) -> dict[str, Any]:
        return generate_dice_counterfactual(
            self.model,
            self.train_df,
            patient,
            self.feature_names,
            self.label_order,
            self.feature_metadata,
            self.dataset,
            task_type=self.task_type,
        )

    def retrieve_guidelines(
        self,
        prediction: dict[str, Any],
        shap_result: dict[str, Any],
        narrative: str | None = None,
    ) -> dict[str, Any]:
        return retrieve_guidelines(
            self.dataset,
            prediction,
            shap_result,
            narrative=narrative,
            top_k=self.retrieval_top_k,
            similarity_threshold=self.retrieval_similarity_threshold,
            vector_index_path=self.vector_index_path,
            embedding_client=self.embedding_client,
        )

    def check_clinical_alignment(
        self, shap_result: dict[str, Any], guideline_context: dict[str, Any]
    ) -> dict[str, Any]:
        return check_clinical_alignment(shap_result, guideline_context)

    def generate_narrative(
        self,
        query: str,
        prediction: dict[str, Any],
        shap_result: dict[str, Any],
        dice_result: dict[str, Any] | None,
        guideline_context: dict[str, Any],
        guardrail_result: dict[str, Any],
    ):
        return generate_narrative_result(
            query,
            prediction,
            shap_result,
            dice_result,
            guideline_context,
            guardrail_result,
            llm_client=self.llm_client,
            fallback_to_template=self.llm_fallback_to_template,
        )
