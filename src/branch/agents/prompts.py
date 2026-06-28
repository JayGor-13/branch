"""Prompt rules used by the deterministic narrative generator."""

NARRATIVE_RULES = [
    "Mention only features present in the SHAP output.",
    "Do not diagnose the patient.",
    "Do not recommend treatment.",
    "Use uncertainty language.",
    "Separate model evidence from any clinical interpretation.",
    "Ground clinical interpretation only in retrieved guideline chunks.",
    "Always include the guardrail status.",
]
