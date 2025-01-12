from src.utilities.common import read_yaml
from src.constants import *
from logger_config import get_logger
import os
from src.entity import DataIngestionConfig, DataPreprocessingConfig, ModelTrainingConfig

logger = get_logger(__name__)

class ConfigurationManager:
    def __init__(self, config_file_path= CONFIG_FILE_PATH):
        self.config= read_yaml(config_file_path)     
        os.makedirs(self.config.artifacts_root, exist_ok=True)
        logger.info("artifact directory created")

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        os.makedirs(config.root_dir, exist_ok = True)
        logger.info("root directory for data_ingestion made")

        return DataIngestionConfig(
            root_dir= config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir  
        )
        

    def get_datapreprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing

        return DataPreprocessingConfig(
            root_dir = config.root_dir,
            source = config.source,
        )

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training

        return ModelTrainingConfig(
            source  = config.source,
            model_save_dir = config.model_save_dir,
            remote_tracking_url = config.remote_tracking_url
        )
