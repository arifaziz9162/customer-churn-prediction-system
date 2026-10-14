import mlflow
from mlflow import MlflowClient

from customer_churn_prediction.entity.config_entity import ModelRegistryConfig
from customer_churn_prediction.logger import ModelRegistryError, get_logger
from customer_churn_prediction.utils.common import (
    load_json,
    save_json,
    save_status,
    setup_mlflow,
)

logger = get_logger("model_registry", "model_registry.log")


class ModelRegistry:
    """Register and promote the final trained model."""

    def __init__(self, config: ModelRegistryConfig):
        """Initialize model registry with configuration."""

        self.config = config

    def initialize_mlflow(self) -> None:
        """Initialize MLflow tracking."""

        try:
            setup_mlflow(self.config.mlflow)
            logger.info("MLflow initialized successfully")

        except Exception as e:
            logger.error(f"MLflow initialization failed: {e}", exc_info=True)
            raise ModelRegistryError("Failed to initialize MLflow") from e

    def load_metrics(self) -> dict:
        """Load model evaluation metrics."""

        try:
            metrics = load_json(self.config.metrics_path)

            if self.config.promotion_metric not in metrics:
                raise KeyError(
                    f"Promotion metric '{self.config.promotion_metric}' "
                    "not found in evaluation metrics"
                )

            logger.info(f"Evaluation metrics loaded: '{metrics}'")
            return metrics

        except Exception as e:
            logger.error(f"Metrics loading failed: {e}", exc_info=True)
            raise ModelRegistryError("Failed to load evaluation metrics") from e

    def load_summary(self) -> dict:
        """Load model evaluation summary."""

        try:
            summary = load_json(self.config.summary_path)

            logger.info("Model evaluation summary loaded successfully")
            return summary

        except Exception as e:
            logger.error(f"Summary loading failed: {e}", exc_info=True)
            raise ModelRegistryError("Failed to load evaluation summary") from e

    def register_model(self, summary: dict) -> str:
        """Register the trained model in MLflow."""

        try:
            model_name = self.config.model_name
            model_uri = summary["model_uri"]

            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
            )

            version = str(registered_model.version)

            logger.info(
                f"Model registered successfully - "
                f"name='{model_name}' version='{version}'"
            )

            return version

        except Exception as e:
            logger.error(f"Model registration failed: {e}", exc_info=True)
            raise ModelRegistryError("Failed to register model") from e

    def get_latest_version(self) -> str:
        """Get the latest registered model version."""

        try:
            client = MlflowClient()

            versions = client.search_model_versions(
                filter_string=f"name='{self.config.model_name}'"
            )

            if not versions:
                raise ValueError("No registered model versions found")

            latest_version = max(
                versions,
                key=lambda model_version: int(model_version.version),
            )

            logger.info(f"Latest model version: '{latest_version.version}'")

            return str(latest_version.version)

        except Exception as e:
            logger.error(f"Failed to get latest model version: {e}", exc_info=True)
            raise ModelRegistryError("Failed to get latest model version") from e

    def get_current_production_version(self) -> str | None:
        """Get the current production model version."""

        try:
            client = MlflowClient()

            model = client.get_registered_model(self.config.model_name)

            version = model.aliases.get(self.config.production_alias)

            if version is None:
                logger.info("No existing Production model found.")
                return None

            logger.info(f"Current production model version: '{version}'")

            return str(version)

        except Exception as e:
            logger.error(
                f"Failed to get current production version: {e}",
                exc_info=True,
            )
            return None

    def get_current_production_score(
        self,
        version: str,
    ) -> float | None:
        """Get the score of the current production model."""

        try:
            client = MlflowClient()

            model_version = client.get_model_version(
                self.config.model_name,
                version,
            )

            run = client.get_run(model_version.run_id)

            metric_name = f"test_{self.config.promotion_metric}"
            score = run.data.metrics.get(metric_name)

            logger.info(
                f"Current production model - "
                f"version='{version}' "
                f"{metric_name}='{score}'"
            )

            return score

        except Exception as e:
            logger.error(
                f"Failed to get current production score: {e}",
                exc_info=True,
            )
            return None

    def promote_model(
        self,
        version: str,
        challenger_score: float,
        champion_version: str | None,
        champion_score: float | None,
        threshold: float,
    ) -> bool:
        """Promote the model if it improves over the production model."""

        try:
            client = MlflowClient()

            model_name = self.config.model_name
            alias = self.config.production_alias
            metric = self.config.promotion_metric

            logger.info(f"Challenger {metric}: '{challenger_score:.4f}'")
            logger.info(f"Champion {metric}: '{champion_score}'")

            if champion_version is None:
                promote = True
                reason = "No production model exists yet."

            elif champion_score is None:
                promote = True
                reason = (
                    f"Production model v{champion_version} has no test_{metric} metric."
                )

            elif challenger_score > champion_score + self.config.min_improvement:
                promote = True
                reason = (
                    f"Challenger {metric} {challenger_score:.4f} "
                    f"beats production {champion_score:.4f}."
                )

            else:
                promote = False
                reason = (
                    f"Challenger {metric} {challenger_score:.4f} "
                    f"does not beat production {champion_score:.4f}."
                )

            client.set_model_version_tag(
                model_name, version, "classification_threshold", str(threshold)
            )

            client.set_model_version_tag(model_name, version, "promoted", str(promote))

            if promote:
                client.set_registered_model_alias(model_name, alias, version)

                logger.info(f"Model version '{version}' promoted to PRODUCTION")

            else:
                logger.info(f"Model version '{version}' was not promoted")

            logger.info(reason)

            return promote

        except Exception as e:
            logger.error(
                f"Model promotion failed: {e}",
                exc_info=True,
            )
            raise ModelRegistryError("Failed to promote model") from e

    def run(self) -> dict:
        """Run the model registry process."""

        try:
            self.initialize_mlflow()

            metrics = self.load_metrics()
            summary = self.load_summary()

            promotion_metric = self.config.promotion_metric
            challenger_score = float(metrics[promotion_metric])
            threshold = float(metrics["threshold"])

            # Register the current model.
            version = self.register_model(summary)

            logger.info(f"MLflow model registered - version='{version}'")

            # Get the current production model.
            champion_version = self.get_current_production_version()

            champion_score = None

            if champion_version is not None:
                champion_score = self.get_current_production_score(champion_version)

            # Promote the exact version registered in this run.
            promoted = self.promote_model(
                version=version,
                challenger_score=challenger_score,
                champion_version=champion_version,
                champion_score=champion_score,
                threshold=threshold,
            )

            report = {
                "model_name": self.config.model_name,
                "challenger_version": version,
                "champion_version": version if promoted else champion_version,
                "promotion_metric": promotion_metric,
                "challenger_score": challenger_score,
                "champion_score": (challenger_score if promoted else champion_score),
                "promoted": promoted,
            }

            save_json(self.config.report_path, report)
            save_status(self.config.status_file, "model_registry", True)

            logger.info(
                f"Model registry process completed successfully - "
                f"version='{version}', promoted='{promoted}'"
            )

            return report

        except Exception as e:
            try:
                save_status(self.config.status_file, "model_registry", False, str(e))
            except Exception:
                logger.exception("Failed to save model registry status")
            logger.error(f"Model registry process failed: {e}", exc_info=True)
            raise ModelRegistryError("Model registry process failed") from e
