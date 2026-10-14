"""Prometheus metrics for customer churn input features."""

from prometheus_client import Histogram

FEATURE_METRICS = {
    "tenure": Histogram(
        "customer_churn_feature_tenure_months",
        "Distribution of customer tenure in months.",
        buckets=(0, 3, 6, 12, 24, 36, 48, 60, 72, 100),
    ),
    "monthly_charges": Histogram(
        "customer_churn_feature_monthly_charges",
        "Distribution of customer monthly charges.",
        buckets=(0, 20, 40, 60, 80, 100, 120, 150, 200),
    ),
    "total_charges": Histogram(
        "customer_churn_feature_total_charges",
        "Distribution of customer total charges.",
        buckets=(0, 100, 500, 1000, 2000, 3000, 5000, 7500, 10000),
    ),
}


def observe_features(features: dict) -> None:
    """Observe numeric features for aggregate monitoring."""
    for feature_name, metric in FEATURE_METRICS.items():
        value = features.get(feature_name)

        if value is None:
            continue

        try:
            metric.observe(float(value))
        except (TypeError, ValueError):
            # Ignore values that cannot be interpreted as numbers.
            continue
