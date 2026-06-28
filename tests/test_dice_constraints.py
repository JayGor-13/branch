from branch.explainability.dice_counterfactual import validate_counterfactual_changes


def test_immutable_age_change_is_invalid():
    metadata = {
        "features": {
            "Age": {"mutable": False, "permitted_range": [10, 70]},
            "Systolic BP": {"mutable": True, "permitted_range": [70, 180]},
        }
    }
    status, problems = validate_counterfactual_changes(
        {"Age": 30, "Systolic BP": 150},
        {"Age": 31, "Systolic BP": 135},
        metadata,
    )
    assert status == "invalid"
    assert problems == ["Immutable feature changed: Age"]
