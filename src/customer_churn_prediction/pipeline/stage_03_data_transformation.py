from customer_churn_prediction.components import DataTransformation
from customer_churn_prediction.config import ConfigurationManager
from customer_churn_prediction.logger import get_logger

logger = get_logger("data_transformation", "data_transformation.log")
STAGE_NAME = "Data Tranformation Stage"


class DataTransformationPipeline:
    """Pipeline for transfoming training data."""

    def run(self):
        """Execute data transformation pipeline."""

        try:
            logger.info(f">>>>>>>>>> {STAGE_NAME} Started <<<<<<<<<<")
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.run()
            logger.info(f">>>>>>>>>> {STAGE_NAME} Completed <<<<<<<<<<")

        except Exception as e:
            logger.error(f"{STAGE_NAME} failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    pipeline = DataTransformationPipeline()
    pipeline.run()
