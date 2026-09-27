"""Smoke test: the Streamlit dashboard runs without exceptions.

Skipped automatically if models have not been trained yet
(run `python src/train_models.py` first).
"""

import pytest

from src.config import ALL_MODELS_PATH, EVALUATION_PATH, PRIMARY_MODEL_PATH, PROJECT_ROOT

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (PRIMARY_MODEL_PATH, ALL_MODELS_PATH, EVALUATION_PATH)),
    reason="Trained models not found - run `python src/train_models.py` first.",
)


def test_dashboard_renders_and_predicts():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Total customers" in labels and "High-risk customers" in labels

    next(b for b in at.button if b.label == "Predict churn risk").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Churn probability" in [m.label for m in at.metric]
