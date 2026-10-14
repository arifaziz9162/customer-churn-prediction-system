"""Unit tests for model evaluation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_churn_prediction.components.model_evaluation import (
    ModelEvaluation,
)
from customer_churn_prediction.logger import ModelEvaluationError


@pytest.fixture
def config(tmp_path):
    config = MagicMock()

    config.target_column = "churn"
    config.model_path = tmp_path / "model.joblib"
    config.test_data_path = tmp_path / "test.csv"
    config.metrics_path = tmp_path / "metrics.json"

    config.cost_per_missed_churn = 100.0
    config.cost_per_unnecessary_offer = 10.0

    config.importance_scoring = "roc_auc"
    config.importance_repeats = 2
    config.random_state = 42
    config.n_jobs = 1

    return config


@pytest.fixture
def evaluator(config):
    return ModelEvaluation(config)


@pytest.fixture
def test_data():
    x = pd.DataFrame(
        {
            "tenure": [1, 2, 24, 36, 48, 60],
            "monthly_charges": [90, 85, 70, 60, 50, 40],
        }
    )

    y = pd.Series([1, 1, 1, 0, 0, 0], name="churn")

    return x, y


def test_prepare_test_data_splits_target(
    evaluator,
    test_data,
):
    x, y = test_data
    frame = x.copy()
    frame["churn"] = y

    features, target = evaluator.prepare_test_data(frame)

    assert "churn" not in features.columns
    assert target.name == "churn"
    assert len(features) == len(target)


def test_prepare_test_data_rejects_missing_target(
    evaluator,
    test_data,
):
    x, _ = test_data

    with pytest.raises(ModelEvaluationError):
        evaluator.prepare_test_data(x)


def test_load_model_wraps_load_error(evaluator):
    with (
        patch(
            "customer_churn_prediction.components.model_evaluation.load_bin",
            side_effect=OSError("missing model"),
        ),
        pytest.raises(ModelEvaluationError),
    ):
        evaluator.load_model()


def test_load_test_data_wraps_read_error(evaluator):
    with (
        patch(
            "customer_churn_prediction.components.model_evaluation.pd.read_csv",
            side_effect=OSError("missing test file"),
        ),
        pytest.raises(ModelEvaluationError),
    ):
        evaluator.load_test_data()


def test_evaluate_returns_metrics_and_predictions(
    evaluator,
    test_data,
    tmp_path,
):
    from sklearn.ensemble import RandomForestClassifier

    x, y = test_data

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )
    model.fit(x, y)

    evaluator.config.metrics_path = tmp_path / "metrics.json"

    with patch(
        "customer_churn_prediction.components.model_evaluation.classification_metrics",
        return_value={
            "accuracy": 0.8,
            "precision": 0.75,
            "recall": 0.8,
            "f1": 0.77,
            "estimated_total_cost": 50.0,
        },
    ):
        metrics, predictions = evaluator.evaluate(
            model,
            x,
            y,
            0.5,
        )

    assert metrics["threshold"] == 0.5
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert len(predictions) == len(y)
    assert evaluator.config.metrics_path.is_file()


def test_evaluate_wraps_prediction_errors(
    evaluator,
    test_data,
):
    x, y = test_data
    model = MagicMock()
    model.predict_proba.side_effect = ValueError("inference failed")

    with pytest.raises(ModelEvaluationError):
        evaluator.evaluate(model, x, y, 0.5)
