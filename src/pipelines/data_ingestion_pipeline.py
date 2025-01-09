from src.config import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from logger_config import get_logger

logger = get_logger(__name__)

STAGE_NAME = "DATA INGESTION"

class DataIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__ == '__main__':
    try:
        logger.info(f"{STAGE_NAME} started")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f"{STAGE_NAME} completed")

    except Exception as e:
        logger.exception(e)
        raise e
