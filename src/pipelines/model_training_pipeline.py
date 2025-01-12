from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from logger_config import get_logger
from src.config import ConfigurationManager
from src.components.model_training import ModelTraining

logger = get_logger(__name__)

STAGE_NAME = "MODEL TRAINING"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(config=model_training_config)
        
        models = {"Linear_Regression": LinearRegression(),
              # "Random_Forest":RandomForestRegressor(),
              # "Gradient_Boosting": GradientBoostingRegressor()}
                 }

        param_grids = {"Linear_Regression": {"fit_intercept": [True, False]},
                      # "Random_Forest": {"n_estimators": [100, 200],
                      #                   "max_depth": [100, 200],
                      #                   "min_samples_split": [2, 5]},
                      # "Gradient_Boosting": {"n_estimators": [100, 200],
                      #                       "learning_rate": [0.1, 0.01],
                      #                       "max_depth": [3, 5]}}
                     }                
        best_models = model_training.train_model(models, param_grids)
        print(best_models)



if __name__ == '__main__':
    try:
        logger.info(f"{STAGE_NAME} started")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f"{STAGE_NAME} completed")

    except Exception as e:
        logger.exception(e)
        raise e
