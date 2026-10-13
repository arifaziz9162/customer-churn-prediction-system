from customer_churn_prediction.components import ModelTrainer
from customer_churn_prediction.config import ConfigurationManager
from customer_churn_prediction.logger import get_logger

logger = get_logger("model_trainer_pipeline", "pipeline.log")
STAGE_NAME = "Model Trainer Stage"


class ModelTrainingPipeline:
    """Pipeline for training machine learning model."""

    def run(self):
        """Execute model trainer pipeline."""

        try:
            logger.info(f">>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<")
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.run()
            logger.info(f">>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<")

        except Exception as e:
            logger.error(f"{STAGE_NAME} failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    pipeline = ModelTrainingPipeline()
    pipeline.run()
