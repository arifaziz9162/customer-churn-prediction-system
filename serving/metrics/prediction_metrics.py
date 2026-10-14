"""Prometheus metrics for customer churn predictions."""

from prometheus_client import Counter, Histogram

PREDICTIONS_TOTAL = Counter(
    "customer_churn_predictions_total",
    "Total number of successful customer churn predictions.",
    ["churn_label"],
)

PREDICTION_ERRORS_TOTAL = Counter(
    "customer_churn_prediction_errors_total",
    "Total number of prediction and persistence submission errors.",
    ["stage"],
)

PREDICTION_PROBABILITY = Histogram(
    "customer_churn_prediction_probability",
    "Distribution of predicted customer churn probabilities.",
    buckets=(
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ),
)

DB_SAVE_TOTAL = Counter(
    "customer_churn_database_writes_total",
    "Total number of completed prediction database writes.",
    ["status"],
)
