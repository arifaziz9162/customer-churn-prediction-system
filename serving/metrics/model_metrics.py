"""Prometheus metrics describing the deployed churn model."""

from prometheus_client import Gauge


MODEL_INFO = Gauge(
    "customer_churn_model_info",
    "Information about the model loaded by the serving application.",
    ["bento_version", "mlflow_version"],
)

MODEL_ACCURACY = Gauge(
    "customer_churn_model_accuracy",
    "Recorded evaluation accuracy of the deployed model.",
)


def publish_model_metrics(
    bento_version: str,
    mlflow_version: str | None,
    accuracy: float | None,
) -> None:
    """Publish metadata and evaluation metrics for the loaded model."""

    # Publish the metadata for the currently loaded model.
    MODEL_INFO.labels(
        bento_version=bento_version or "unknown",
        mlflow_version=mlflow_version or "unknown",
    ).set(1)

    # Publish the model's recorded evaluation accuracy.
    MODEL_ACCURACY.set(float(accuracy) if accuracy is not None else 0.0)
