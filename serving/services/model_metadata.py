"""Metadata associated with the deployed customer churn model."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelMetadata:
    """Store information about the model currently being served."""

    name: str
    bento_version: str
    mlflow_version: str | None
    mlflow_run_id: str | None
    classification_threshold: float
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    roc_auc: float | None = None

    @property
    def display_version(self) -> str:
        """Prefer the MLflow registry version for display."""
        return self.mlflow_version or self.bento_version

    @classmethod
    def from_bento_model(cls, bento_model: Any) -> "ModelMetadata":
        """Build metadata from a BentoML model reference."""
        info = bento_model.info
        labels = info.labels or {}
        metadata = info.metadata or {}

        metrics = metadata.get("test_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        registry_version = (
            labels.get("registry_version")
            or metadata.get("registry_version")
            or labels.get("mlflow_version")
        )

        return cls(
            name=bento_model.tag.name,
            bento_version=bento_model.tag.version,
            mlflow_version=(
                str(registry_version)
                if registry_version is not None
                else None
            ),
            mlflow_run_id=metadata.get("mlflow_run_id"),
            classification_threshold=float(
                metadata.get("classification_threshold", 0.3)
            ),
            accuracy=metrics.get("accuracy"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1_score=metrics.get("f1", metrics.get("f1_score")),
            roc_auc=metrics.get("roc_auc"),
        )
