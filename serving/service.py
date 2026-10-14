"""BentoML service for customer churn predictions."""

import bentoml

from customer_churn_prediction.logger import get_logger
from serving.config import settings
from serving.metrics import publish_model_metrics
from serving.persistence import PredictionRepository
from serving.schemas import (
    CustomerChurnInput,
    ModelInfoResponse,
    PredictionResponse,
    PredictionsRequest,
    PredictionsResponse,
)
from serving.services import ModelMetadata, PredictionService

logger = get_logger("service", "service.log")


@bentoml.service(
    name=settings.SERVICE_NAME,
    traffic={
        "timeout": settings.REQUEST_TIMEOUT_SECONDS,
        "concurrency": settings.MAX_CONCURRENCY,
    },
)
class CustomerChurnService:
    """Serve customer churn predictions using a packaged model."""

    # Resolve the exact model artifact configured for this deployment.
    bento_model = bentoml.models.BentoModel(settings.BENTO_MODEL_TAG)

    def __init__(self) -> None:
        """Load the model and initialize the serving components."""

        # Load the model through BentoML's model-loading interface.
        # This avoids assuming the model artifact is a standalone joblib file.
        model = bentoml.sklearn.load_model(self.bento_model.tag)

        # Read version, threshold, and evaluation metrics from model metadata.
        metadata = ModelMetadata.from_bento_model(self.bento_model)

        # Enable database persistence only when configured.
        repository = PredictionRepository() if settings.DB_SAVE_ENABLED else None

        self.prediction_service = PredictionService(
            model=model,
            metadata=metadata,
            repository=repository,
        )

        publish_model_metrics(
            bento_version=metadata.bento_version,
            mlflow_version=metadata.mlflow_version,
            accuracy=metadata.accuracy,
        )

        logger.info(
            "Customer churn service initialized successfully. "
            "Model=%s, BentoML version=%s, MLflow version=%s",
            metadata.name,
            metadata.bento_version,
            metadata.mlflow_version or "unknown",
        )

    @bentoml.api(
        route="/api/v1/predict",
        input_spec=CustomerChurnInput,
        output_spec=PredictionResponse,
    )
    def predict(self, **params) -> PredictionResponse:
        """Predict whether a customer is likely to churn."""
        customer = CustomerChurnInput(**params)

        return self.prediction_service.predict(customer)

    @bentoml.api(
        route="/api/v1/model_info",
        output_spec=ModelInfoResponse,
    )
    def model_info(self) -> ModelInfoResponse:
        """Return metadata for the currently served model."""
        return self.prediction_service.model_info()

    @bentoml.api(
        route="/api/v1/predictions",
        input_spec=PredictionsRequest,
        output_spec=PredictionsResponse,
    )
    def stored_predictions(
        self,
        **params,
    ) -> PredictionsResponse:
        """Return recent persisted prediction history."""
        request = PredictionsRequest(**params)

        return self.prediction_service.recent_predictions(request.limit)
