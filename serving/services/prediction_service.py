"""Business logic for customer churn predictions."""

import pandas as pd

from customer_churn_prediction.logger import (
    ModelInferenceError,
    get_logger,
)
from serving.config import settings
from serving.constants import FEATURE_COLUMNS
from serving.metrics import (
    PREDICTION_ERRORS_TOTAL,
    PREDICTION_PROBABILITY,
    PREDICTIONS_TOTAL,
    observe_features,
)
from serving.persistence import PredictionRepository
from serving.schemas import (
    CustomerChurnInput,
    ModelInfoResponse,
    PredictionResponse,
    PredictionsResponse,
)
from serving.services.model_metadata import ModelMetadata
from serving.services.quality_label import get_churn_label

logger = get_logger("prediction_service", "serving.log")


class PredictionService:
    """Generate churn predictions and record serving metrics."""

    def __init__(
        self,
        model,
        metadata: ModelMetadata,
        repository: PredictionRepository | None = None,
    ) -> None:
        """Initialize the model, metadata, and optional repository."""
        self._model = model
        self._metadata = metadata
        self._repository = repository

        self._validate_model_features()

    def _validate_model_features(self) -> None:
        """Ensure serving features match the model's expected columns."""
        expected = getattr(self._model, "feature_names_in_", None)

        if expected is not None and tuple(expected) != FEATURE_COLUMNS:
            raise ModelInferenceError(
                "Model feature names or order do not match the serving "
                f"configuration. Model={tuple(expected)}, "
                f"serving={FEATURE_COLUMNS}"
            )

        expected_count = getattr(self._model, "n_features_in_", None)

        if (
            expected_count is not None
            and expected_count != len(FEATURE_COLUMNS)
        ):
            raise ModelInferenceError(
                "Model feature count does not match the serving "
                f"configuration. Model={expected_count}, "
                f"serving={len(FEATURE_COLUMNS)}"
            )

    @staticmethod
    def build_features(data: CustomerChurnInput) -> pd.DataFrame:
        """Build a single-row DataFrame in training feature order."""
        values = data.model_dump()

        missing = set(FEATURE_COLUMNS) - values.keys()
        if missing:
            raise ValueError(
                f"Missing required model features: {sorted(missing)}"
            )

        return pd.DataFrame(
            [[values[column] for column in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )

    def predict(self, data: CustomerChurnInput) -> PredictionResponse:
        """Predict churn probability and apply the configured threshold."""
        try:
            features = self.build_features(data)

            probabilities = self._model.predict_proba(features)
            classes = list(self._model.classes_)

            if 1 not in classes:
                raise ValueError(
                    "The trained model does not contain positive class 1."
                )

            positive_class_index = classes.index(1)
            probability = float(
                probabilities[0][positive_class_index]
            )

            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    "The model returned an invalid probability."
                )

            threshold = self._metadata.classification_threshold
            prediction = int(probability >= threshold)

        except Exception as exc:
            PREDICTION_ERRORS_TOTAL.labels(
                stage="inference"
            ).inc()

            logger.error(
                "Churn prediction failed.",
                exc_info=True,
            )

            raise ModelInferenceError(
                "Churn prediction failed."
            ) from exc

        label = get_churn_label(prediction)

        PREDICTIONS_TOTAL.labels(
            churn_label=label.value
        ).inc()

        PREDICTION_PROBABILITY.observe(probability)
        observe_features(data.model_dump())

        logger.info(
            "Churn prediction completed: label=%s probability=%.4f",
            label.value,
            probability,
        )

        if self._repository is not None:
            try:
                self._repository.save_async(
                    features.iloc[0].tolist(),
                    prediction,
                    probability,
                )
            except Exception:
                PREDICTION_ERRORS_TOTAL.labels(
                    stage="persistence"
                ).inc()

                logger.exception(
                    "Failed to submit prediction for persistence."
                )

        return PredictionResponse(
            churn_probability=probability,
            churn_prediction=prediction,
            churn_label=label,
            classification_threshold=threshold,
            message=(
                "Customer is predicted to churn."
                if prediction == 1
                else "Customer is predicted not to churn."
            ),
            model_version=self._metadata.display_version,
        )

    def model_info(self) -> ModelInfoResponse:
        """Return metadata about the currently deployed model."""
        meta = self._metadata

        return ModelInfoResponse(
            status="Running",
            service=settings.SERVICE_NAME,
            service_version=settings.APP_VERSION,
            bento_model=f"{meta.name}:{meta.bento_version}",
            mlflow_version=meta.mlflow_version or "unknown",
            mlflow_run_id=meta.mlflow_run_id,
            classification_threshold=meta.classification_threshold,
            accuracy=meta.accuracy,
            precision=meta.precision,
            recall=meta.recall,
            f1_score=meta.f1_score,
            roc_auc=meta.roc_auc,
        )

    def recent_predictions(self, limit: int) -> PredictionsResponse:
        """Return prediction history when persistence is enabled."""
        if self._repository is None:
            return PredictionsResponse(
                count=0,
                predictions=[],
            )

        safe_limit = max(
            1,
            min(limit, settings.PREDICTIONS_MAX_LIMIT),
        )

        rows = self._repository.fetch_recent(safe_limit)

        return PredictionsResponse(
            count=len(rows),
            predictions=rows,
        )
