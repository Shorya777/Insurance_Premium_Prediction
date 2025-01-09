import pandas as pd
from src.utilities.common import custom_grid_search_models
from src.entity import ModelTrainingConfig
from logger_config import get_logger

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        
    def train_model(self, models, param_grid):
        train_df = pd.read_csv(self.config.source)
        logger.info("Converting Training Data boolean values to integer")
        for col in train_df.columns:
            if train_df[col].dtype == 'bool':
                train_df[col] = train_df[col].astype(int)
        logger.info("Boolean values converted to Integer")
        
        X_train = train_df.drop(["Premium_Amount"], axis = 1).to_numpy()
        y_train = train_df["Premium_Amount"].to_numpy()
        logger.info("Training Process initiated")
        best_models = custom_grid_search_models(models = models, save_dir= self.config.model_save_dir, 
                                                param_grid= param_grid, X_train= X_train, y_train= y_train, cv= 2)
        logger.info("Training Process done successfully")

        return best_models
