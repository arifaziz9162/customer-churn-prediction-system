"""Tests for request and response schemas."""

import pytest
from pydantic import ValidationError

from serving.schemas.request_schema import CustomerChurnInput
from serving.schemas.response_schema import PredictionResponse


@pytest.fixture
def valid_customer_payload():
    """Return a valid customer payload for prediction."""
    return {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 70.0,
        "total_charges": 850.0,
    }


def test_customer_churn_input_accepts_valid_payload(valid_customer_payload):
    """Accept a complete, valid customer payload."""
    customer = CustomerChurnInput(**valid_customer_payload)

    assert customer.gender == "Female"
    assert customer.tenure == 12
    assert customer.monthly_charges == 70.0


def test_customer_churn_input_rejects_missing_field(valid_customer_payload):
    """Reject payloads missing required features."""
    valid_customer_payload.pop("tenure")

    with pytest.raises(ValidationError):
        CustomerChurnInput(**valid_customer_payload)


def test_customer_churn_input_rejects_extra_field(valid_customer_payload):
    """Reject unexpected input fields."""
    valid_customer_payload["unexpected_feature"] = "value"

    with pytest.raises(ValidationError):
        CustomerChurnInput(**valid_customer_payload)


def test_customer_churn_input_rejects_invalid_numeric_value(
    valid_customer_payload,
):
    """Reject an invalid value for a numeric feature."""
    valid_customer_payload["monthly_charges"] = "not-a-number"

    with pytest.raises(ValidationError):
        CustomerChurnInput(**valid_customer_payload)


def test_prediction_response_accepts_valid_data():
    """Accept a valid prediction response."""
    response = PredictionResponse(
        churn_probability=0.5853,
        churn_prediction=1,
        churn_label="Yes",
        classification_threshold=0.3,
        message="Customer is predicted to churn.",
        model_version="2",
    )

    assert response.churn_label == "Yes"
    assert response.churn_prediction == 1
    assert response.churn_probability == pytest.approx(0.5853)


def test_prediction_response_serializes_to_json():
    """Serialize a prediction response into JSON-compatible data."""
    response = PredictionResponse(
        churn_probability=0.2,
        churn_prediction=0,
        churn_label="No",
        classification_threshold=0.3,
        message="Customer is not predicted to churn.",
        model_version="2",
    )

    data = response.model_dump(mode="json")

    assert data["churn_label"] == "No"
    assert data["churn_prediction"] == 0
