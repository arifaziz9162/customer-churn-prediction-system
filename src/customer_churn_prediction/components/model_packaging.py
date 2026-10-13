import mlflow
from mlflow.tracking import MlflowClient

from customer_churn_prediction.entity.config_entity import ModelPackagingConfig
from customer_churn_prediction.logger import ModelPackagingError, get_logger
from customer_churn_prediction.utils.common import save_json, save_status, setup_mlflow


logger = get_logger("model_packaging", "model_packaging.log")


def _import_bentoml():
    """Import BentoML when required."""

    import bentoml  # type: ignore[import-not-found]

    return bentoml


class ModelPackaging:
    """Turn the current Production model into a versioned BentoML model."""

    def __init__(self, config: ModelPackagingConfig):
        """Initialize model packaging with configuration."""

        self.config = config

    def initialize_mlflow(self) -> None:
        """Initialize MLflow tracking."""

        try:
            setup_mlflow(self.config.mlflow)
            logger.info("MLflow initialized successfully")

        except Exception as e:
            logger.error(f"MLflow initialization failed: {e}", exc_info=True)
            raise ModelPackagingError("Failed to initialize MLflow") from e

    def get_production_version(self):
        """Get the model version currently assigned to the Production alias."""

        try:
            client = MlflowClient()
            model = client.get_registered_model(self.config.registry_model_name)
            version = model.aliases.get(self.config.production_alias)
            if version is None:
                raise ValueError(
                    f"No '{self.config.registry_model_name}' model "
                    f"found under alias '{self.config.production_alias}'. "
                    "Nothing to package."
                )

            version = client.get_model_version(
                self.config.registry_model_name,
                str(version),
            )

            logger.info(
                f"Found Production model - "
                f"version='{version.version}' "
                f"run_id='{version.run_id}'"
            )

            return version

        except Exception as e:
            logger.error(f"Failed to find Production model: {e}", exc_info=True)
            raise ModelPackagingError("Failed to find Production model") from e

    def get_metrics(self, run_id: str) -> dict:
        """Read and validate evaluation metrics logged for the model run."""

        try:
            client = MlflowClient()
            metrics = client.get_run(run_id).data.metrics

            test_metrics = {
                key.removeprefix("test_"): float(value)
                for key, value in metrics.items()
                if key.startswith("test_")
            }

            if not test_metrics:
                raise ValueError(f"No test evaluation metrics found for run '{run_id}'")

            logger.info(f"Model evaluation metrics loaded: '{test_metrics}'")

            return test_metrics

        except Exception as e:
            logger.error(f"Failed to read run metrics: {e}", exc_info=True)
            raise ModelPackagingError("Failed to read run metrics") from e

    def get_threshold(self, version) -> float:
        """Read the classification threshold from the registered model."""

        try:
            threshold = version.tags.get("classification_threshold")
            if threshold is None:
                raise ValueError("Classification threshold not found in model tags")
            threshold = float(threshold)
            logger.info(f"Classification threshold loaded: '{threshold}'")

            return threshold

        except Exception as e:
            logger.error(
                f"Failed to read classification threshold: {e}",
                exc_info=True,
            )
            raise ModelPackagingError("Failed to read classification threshold") from e

    def load_model(self, version):
        """Load the registered model from MLflow."""

        try:
            model_uri = f"models:/{self.config.registry_model_name}/{version.version}"
            logger.info(f"Loading MLflow model from URI: '{model_uri}'")
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Successfully loaded MLflow model version '{version.version}'")

            return model

        except Exception as e:
            logger.error(
                f"Failed to load model: {e}",
                exc_info=True,
            )
            raise ModelPackagingError("Failed to load model from MLflow") from e

    def find_existing(self, bentoml, registry_version: str):
        """Find an existing BentoML model for the exact MLflow model version."""

        try:
            candidates = bentoml.models.list(self.config.bento_model_name)

        except bentoml.exceptions.NotFound:
            return None

        for existing in candidates:
            metadata = existing.info.metadata or {}

            if (
                metadata.get("registry_model") == self.config.registry_model_name
                and metadata.get("registry_version") == registry_version
            ):
                return existing

        return None

    def package(
        self,
        model,
        version,
        metrics: dict,
        threshold: float,
    ):
        """Save the model into the BentoML model store."""

        try:
            bentoml = _import_bentoml()
            registry_version = str(version.version)
            existing = self.find_existing(
                bentoml,
                registry_version,
            )
            if existing is not None:
                logger.info(
                    f"Registry version '{registry_version}' "
                    f"already packaged as '{existing.tag}', "
                    "reusing it"
                )
                return existing

            labels = {
                "registry_model": self.config.registry_model_name,
                "registry_version": registry_version,
                "alias": self.config.production_alias,
            }
            metadata = {
                "registry_model": self.config.registry_model_name,
                "registry_version": registry_version,
                "mlflow_run_id": version.run_id,
                "classification_threshold": threshold,
                "test_metrics": metrics,
            }
            bento_model = bentoml.sklearn.save_model(
                self.config.bento_model_name,
                model,
                signatures={
                    "predict": {"batchable": True},
                    "predict_proba": {"batchable": True},
                },
                labels=labels,
                metadata=metadata,
            )

            logger.info(
                f"Model packaged successfully - bento_model='{bento_model.tag}'"
            )

            return bento_model

        except ModelPackagingError:
            raise

        except Exception as e:
            logger.error(f"Packaging failed: {e}", exc_info=True)
            raise ModelPackagingError("Failed to package model") from e

    def save_info(
        self,
        bento_model,
        version,
        metrics: dict,
        threshold: float,
    ) -> None:
        """Save packaging information for CI/CD."""

        save_json(
            self.config.report_path,
            {
                "bento_model": str(bento_model.tag),
                "registry_model_name": self.config.registry_model_name,
                "registry_version": str(version.version),
                "mlflow_run_id": version.run_id,
                "classification_threshold": threshold,
                "test_metrics": metrics,
            },
        )

        logger.info(f"Packaging information saved to '{self.config.report_path}'")

    def run(self) -> dict:
        """Run the model packaging process."""

        try:
            self.initialize_mlflow()
            version = self.get_production_version()
            metrics = self.get_metrics(version.run_id)
            threshold = self.get_threshold(version)
            model = self.load_model(version)
            bento_model = self.package(
                model=model,
                version=version,
                metrics=metrics,
                threshold=threshold,
            )
            self.save_info(
                bento_model=bento_model,
                version=version,
                metrics=metrics,
                threshold=threshold,
            )
            report = {
                "bento_model": str(bento_model.tag),
                "registry_model_name": self.config.registry_model_name,
                "registry_version": str(version.version),
                "classification_threshold": threshold,
                "packaged": True,
            }
            save_status(self.config.status_file, "model_packaging", True)
            logger.info("Model packaging completed successfully")

            return report

        except Exception as e:
            try:
                save_status(self.config.status_file, "model_packaging", False, str(e))
            except Exception:
                logger.exception("Failed to save model packaging status")

            logger.error(f"Model packaging process failed: {e}", exc_info=True)
            raise ModelPackagingError("Model packaging process failed") from e
