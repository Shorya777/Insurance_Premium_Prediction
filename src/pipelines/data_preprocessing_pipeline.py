from src.config import ConfigurationManager
from src.components.data_preprocessing import DataPreprocessing
from logger_config import get_logger

logger = get_logger(__name__)

STAGE_NAME = "DATA PREPROCESSING"

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_datapreprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.data_preprocessing_pipeline()
            

if __name__ == '__main__':
    try:
        logger.info(f"{STAGE_NAME} started")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f"{STAGE_NAME} completed")

    except Exception as e:
        logger.exception(e)
        raise e
