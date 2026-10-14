"""Shared constants for customer churn model serving."""

SERVICE_NAME = "customer_churn_service"

API_VERSION = "v1"

PREDICTION_ROUTE = "/api/v1/predict"

MODEL_INFO_ROUTE = "/api/v1/model_info"

PREDICTIONS_ROUTE = "/api/v1/predictions"

# Keep this order consistent with the training pipeline.
FEATURE_COLUMNS: tuple[str, ...] = (
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
)
