"""Unit tests for MLflow model registration and promotion."""

from unittest.mock import MagicMock, patch

import pytest

from customer_churn_prediction.components.model_registry import (
    ModelRegistry,
)
from customer_churn_prediction.logger import ModelRegistryError


@pytest.fixture
def config():
    config = MagicMock()

    config.mlflow = MagicMock()
    config.metrics_path = "metrics.json"
    config.summary_path = "summary.json"
    config.report_path = "report.json"
    config.status_file = "status.json"

    config.model_name = "customer-churn-model"
    config.production_alias = "production"
    config.promotion_metric = "roc_auc"
    config.min_improvement = 0.01

    return config


@pytest.fixture
def registry(config):
    return ModelRegistry(config)


def test_initialize_mlflow(registry):
    with patch(
        "customer_churn_prediction.components.model_registry.setup_mlflow"
    ) as setup:
        registry.initialize_mlflow()

    setup.assert_called_once_with(registry.config.mlflow)


def test_initialize_mlflow_wraps_error(registry):
    with (
        patch(
            "customer_churn_prediction.components.model_registry.setup_mlflow",
            side_effect=RuntimeError("tracking unavailable"),
        ),
        pytest.raises(ModelRegistryError),
    ):
        registry.initialize_mlflow()


def test_load_metrics_returns_metrics(registry):
    expected = {"roc_auc": 0.84, "threshold": 0.3}

    with patch(
        "customer_churn_prediction.components.model_registry.load_json",
        return_value=expected,
    ):
        result = registry.load_metrics()

    assert result == expected


def test_load_metrics_requires_promotion_metric(registry):
    with (
        patch(
            "customer_churn_prediction.components.model_registry.load_json",
            return_value={"accuracy": 0.9},
        ),
        pytest.raises(ModelRegistryError),
    ):
        registry.load_metrics()


def test_load_summary_returns_summary(registry):
    expected = {"run_id": "run-123", "model_uri": "runs:/run-123/model"}

    with patch(
        "customer_churn_prediction.components.model_registry.load_json",
        return_value=expected,
    ):
        assert registry.load_summary() == expected


def test_register_model_returns_version(registry):
    with patch(
        "customer_churn_prediction.components.model_registry.mlflow.register_model",
        return_value=MagicMock(version="3"),
    ) as register:
        result = registry.register_model({"model_uri": "runs:/run-123/model"})

    assert result == "3"
    register.assert_called_once_with(
        model_uri="runs:/run-123/model",
        name=registry.config.model_name,
    )


def test_get_latest_version(registry):
    versions = [
        MagicMock(version="1"),
        MagicMock(version="3"),
        MagicMock(version="2"),
    ]

    client = MagicMock()
    client.search_model_versions.return_value = versions

    with patch(
        "customer_churn_prediction.components.model_registry.MlflowClient",
        return_value=client,
    ):
        result = registry.get_latest_version()

    assert result == "3"


def test_get_latest_version_fails_when_empty(registry):
    client = MagicMock()
    client.search_model_versions.return_value = []

    with (
        patch(
            "customer_churn_prediction.components.model_registry.MlflowClient",
            return_value=client,
        ),
        pytest.raises(ModelRegistryError),
    ):
        registry.get_latest_version()


def test_promote_model_when_no_champion(registry):
    client = MagicMock()

    with patch(
        "customer_churn_prediction.components.model_registry.MlflowClient",
        return_value=client,
    ):
        result = registry.promote_model(
            version="2",
            challenger_score=0.84,
            champion_version=None,
            champion_score=None,
            threshold=0.3,
        )

    assert result is True

    client.set_registered_model_alias.assert_called_once_with(
        registry.config.model_name,
        registry.config.production_alias,
        "2",
    )


def test_do_not_promote_without_improvement(registry):
    client = MagicMock()

    with patch(
        "customer_churn_prediction.components.model_registry.MlflowClient",
        return_value=client,
    ):
        result = registry.promote_model(
            version="3",
            challenger_score=0.84,
            champion_version="2",
            champion_score=0.85,
            threshold=0.3,
        )

    assert result is False
    client.set_registered_model_alias.assert_not_called()


def test_get_current_production_version_returns_none_on_error(
    registry,
):
    with patch(
        "customer_churn_prediction.components.model_registry.MlflowClient",
        side_effect=RuntimeError("tracking unavailable"),
    ):
        assert registry.get_current_production_version() is None
