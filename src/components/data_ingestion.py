import os
import zipfile
import subprocess
from src.entity import DataIngestionConfig
from logger_config import get_logger

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def kaggle_download(self, dataset_name, file):
        command = f"kaggle competitions download -c {dataset_name} -p {file}"
        try:
            subprocess.run(command, shell=True, check=True)
            print("Dataset downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during download: {e}")

    def download_file(self) -> str:
        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.root_dir
            os.makedirs(self.config.root_dir, exist_ok=True)
            
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")
            self.kaggle_download(dataset_url, zip_download_dir)
            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        logger.info(f"{unzip_path} directory made")
        logger.info(self.config.local_data_file)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"data unzipped successfully in directory {unzip_path}")
