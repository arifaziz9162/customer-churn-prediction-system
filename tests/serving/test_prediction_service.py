"""Tests for the prediction service."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from customer_churn_prediction.logger.exception import ModelInferenceError
from serving.schemas.request_schema import CustomerChurnInput
from serving.services.prediction_service import PredictionService


@pytest.fixture
def mock_model():
    """Create a mock classifier returning churn probabilities."""

    model = Mock(spec=["predict_proba", "classes_"])
    model.classes_ = np.array([0, 1])
    model.predict_proba.return_value = np.array([[0.4147, 0.5853]])

    return model


@pytest.fixture
def mock_metadata():
    """Provide model metadata for prediction-service tests."""

    return Mock(
        classification_threshold=0.3,
        display_version="test-model-version",
        bento_version="test-bento-version",
        mlflow_version="2",
        accuracy=0.80,
        precision=0.75,
        recall=0.70,
        f1_score=0.72,
        roc_auc=0.82,
    )


@pytest.fixture
def mock_repository():
    """Create a mock prediction repository."""

    return Mock()


@pytest.fixture
def valid_features():
    """Create a valid customer input model for prediction tests."""

    return CustomerChurnInput(
        gender="Female",
        senior_citizen=0,
        partner="Yes",
        dependents="No",
        tenure=12,
        phone_service="Yes",
        multiple_lines="No",
        internet_service="DSL",
        online_security="Yes",
        online_backup="No",
        device_protection="No",
        tech_support="No",
        streaming_tv="No",
        streaming_movies="No",
        contract="Month-to-month",
        paperless_billing="Yes",
        payment_method="Electronic check",
        monthly_charges=70.0,
        total_charges=850.0,
    )


@pytest.fixture
def prediction_service(mock_model, mock_metadata, mock_repository):
    """Construct a prediction service with mocked dependencies."""

    return PredictionService(
        model=mock_model,
        metadata=mock_metadata,
        repository=mock_repository,
    )


def test_predict_returns_expected_churn_result(
    prediction_service,
    mock_model,
    mock_repository,
    valid_features,
):
    """Predict churn and persist the result asynchronously."""

    result = prediction_service.predict(valid_features)

    assert result.churn_probability == pytest.approx(0.5853)
    assert result.churn_prediction == 1
    assert result.churn_label == "Yes"
    assert result.classification_threshold == 0.3

    mock_model.predict_proba.assert_called_once()
    mock_repository.save_async.assert_called_once()


def test_predict_returns_no_churn_below_threshold(
    prediction_service,
    mock_model,
    valid_features,
):
    """Classify the customer as non-churning below the threshold."""

    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])

    result = prediction_service.predict(valid_features)

    assert result.churn_probability == pytest.approx(0.2)
    assert result.churn_prediction == 0
    assert result.churn_label == "No"


def test_predict_passes_features_in_dataframe(
    prediction_service,
    mock_model,
    valid_features,
):
    """Pass customer features to the model as a DataFrame."""

    prediction_service.predict(valid_features)

    model_input = mock_model.predict_proba.call_args.args[0]

    assert isinstance(model_input, pd.DataFrame)
    assert len(model_input) == 1
    assert model_input.iloc[0]["tenure"] == 12


def test_predict_does_not_save_when_repository_is_disabled(
    mock_model,
    mock_metadata,
    valid_features,
):
    """Support inference without database persistence."""

    service = PredictionService(
        model=mock_model,
        metadata=mock_metadata,
        repository=None,
    )

    result = service.predict(valid_features)

    assert result.churn_label == "Yes"
    mock_model.predict_proba.assert_called_once()


def test_predict_wraps_model_failure(
    prediction_service,
    mock_model,
    valid_features,
):
    """Wrap model inference failures in the service exception."""

    mock_model.predict_proba.side_effect = RuntimeError("Model failed")

    with pytest.raises(
        ModelInferenceError,
        match="Churn prediction failed",
    ):
        prediction_service.predict(valid_features)
