"""Unit tests for BentoML model packaging."""

from unittest.mock import MagicMock, patch

import pytest

from customer_churn_prediction.components.model_packaging import (
    ModelPackaging,
)
from customer_churn_prediction.logger import ModelPackagingError


@pytest.fixture
def config():
    config = MagicMock()

    config.mlflow = MagicMock()
    config.registry_model_name = "customer-churn-model"
    config.production_alias = "production"
    config.bento_model_name = "customer_churn_model"

    config.report_path = "packaging_report.json"
    config.status_file = "status.json"

    return config


@pytest.fixture
def packaging(config):
    return ModelPackaging(config)


def test_initialize_mlflow(packaging):
    with patch(
        "customer_churn_prediction.components.model_packaging.setup_mlflow"
    ) as setup:
        packaging.initialize_mlflow()

    setup.assert_called_once_with(packaging.config.mlflow)


def test_get_production_version(packaging):
    model_version = MagicMock()
    model_version.version = "2"
    model_version.run_id = "run-123"

    registered_model = MagicMock()
    registered_model.aliases = {"production": "2"}

    client = MagicMock()
    client.get_registered_model.return_value = registered_model
    client.get_model_version.return_value = model_version

    with patch(
        "customer_churn_prediction.components.model_packaging.MlflowClient",
        return_value=client,
    ):
        result = packaging.get_production_version()

    assert result.version == "2"
    client.get_model_version.assert_called_once_with(
        packaging.config.registry_model_name,
        "2",
    )


def test_get_production_version_fails_without_alias(
    packaging,
):
    registered_model = MagicMock()
    registered_model.aliases = {}

    client = MagicMock()
    client.get_registered_model.return_value = registered_model

    with (
        patch(
            "customer_churn_prediction.components.model_packaging.MlflowClient",
            return_value=client,
        ),
        pytest.raises(ModelPackagingError),
    ):
        packaging.get_production_version()


def test_get_metrics_extracts_test_metrics(packaging):
    run = MagicMock()
    run.data.metrics = {
        "test_accuracy": 0.81,
        "test_roc_auc": 0.84,
        "train_accuracy": 0.90,
    }

    client = MagicMock()
    client.get_run.return_value = run

    with patch(
        "customer_churn_prediction.components.model_packaging.MlflowClient",
        return_value=client,
    ):
        metrics = packaging.get_metrics("run-123")

    assert metrics == {
        "accuracy": 0.81,
        "roc_auc": 0.84,
    }


def test_get_metrics_rejects_missing_test_metrics(packaging):
    run = MagicMock()
    run.data.metrics = {"train_accuracy": 0.9}

    client = MagicMock()
    client.get_run.return_value = run

    with (
        patch(
            "customer_churn_prediction.components.model_packaging.MlflowClient",
            return_value=client,
        ),
        pytest.raises(ModelPackagingError),
    ):
        packaging.get_metrics("run-123")


def test_get_threshold_reads_model_tag(packaging):
    version = MagicMock()
    version.tags = {"classification_threshold": "0.3"}

    assert packaging.get_threshold(version) == 0.3


def test_get_threshold_rejects_missing_tag(packaging):
    version = MagicMock()
    version.tags = {}

    with pytest.raises(ModelPackagingError):
        packaging.get_threshold(version)


def test_load_model(packaging):
    """Test loading a model without accessing the real MLflow registry."""
    version = MagicMock()
    version.version = "2"

    expected_model = MagicMock()
    model_uri = f"models:/{packaging.config.registry_model_name}/2"

    with patch.object(
        packaging,
        "load_model",
        return_value=expected_model,
    ) as mocked_load_model:
        result = mocked_load_model(version)

    assert result is expected_model
    mocked_load_model.assert_called_once_with(version)


def test_package_reuses_existing_model(packaging):
    existing = MagicMock()
    bentoml = MagicMock()
    bentoml.exceptions.NotFound = type("NotFound", (Exception,), {})

    with (
        patch(
            "customer_churn_prediction.components.model_packaging._import_bentoml",
            return_value=bentoml,
        ),
        patch.object(
            packaging,
            "find_existing",
            return_value=existing,
        ),
    ):
        result = packaging.package(
            model=MagicMock(),
            version=MagicMock(version="2", run_id="run-123"),
            metrics={"accuracy": 0.8},
            threshold=0.3,
        )

    assert result is existing
    bentoml.sklearn.save_model.assert_not_called()


def test_package_saves_new_model(packaging):
    bentoml = MagicMock()
    bentoml.exceptions.NotFound = type("NotFound", (Exception,), {})

    expected = MagicMock()
    bentoml.sklearn.save_model.return_value = expected

    version = MagicMock(version="2", run_id="run-123")

    with (
        patch(
            "customer_churn_prediction.components.model_packaging._import_bentoml",
            return_value=bentoml,
        ),
        patch.object(
            packaging,
            "find_existing",
            return_value=None,
        ),
    ):
        result = packaging.package(
            model=MagicMock(),
            version=version,
            metrics={"accuracy": 0.8},
            threshold=0.3,
        )

    assert result is expected
    bentoml.sklearn.save_model.assert_called_once()
