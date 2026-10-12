from customer_churn_prediction.components import DataValidation
from customer_churn_prediction.config import ConfigurationManager
from customer_churn_prediction.logger import get_logger

logger = get_logger("data_validation", "data_validation.log")
STAGE_NAME = "Data Validation Stage"


class DataValidationPipeline:
    """Pipeline for validating training data."""

    def run(self):
        """Execute data validation pipeline."""

        try:
            logger.info(f">>>>>>> {STAGE_NAME} Started <<<<<<<")
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation.run()
            logger.info(f">>>>>>> {STAGE_NAME} Completed <<<<<<<")

        except Exception as e:
            logger.error(f"{STAGE_NAME} failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    pipeline = DataValidationPipeline()
    pipeline.run()
