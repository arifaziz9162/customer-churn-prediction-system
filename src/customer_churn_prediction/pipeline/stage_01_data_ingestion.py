from customer_churn_prediction.components import DataIngestion
from customer_churn_prediction.config import ConfigurationManager
from customer_churn_prediction.logger import get_logger

logger = get_logger("data_ingestion", "data_ingestion.log")
STAGE_NAME = "Data Ingestion Stage"


class DataIngestionPipeline:
    """Pipeline for ingestion training data."""

    def run(self):
        """Execute data ingestion pipeline."""

        try:
            logger.info(f">>>>>>> {STAGE_NAME} Started <<<<<<<")
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.run()
            logger.info(f">>>>>>> {STAGE_NAME} Completed <<<<<<<")

        except Exception as e:
            logger.error(f"{STAGE_NAME} failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    pipeline.run()
