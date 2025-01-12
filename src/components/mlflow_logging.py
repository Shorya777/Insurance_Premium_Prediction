from sklearn.model_selection import GridSearchCV
import mlflow
import os
import joblib
import dagshub

class Logger:
    def __init__(self):
        self.cv_results = None  # Placeholder for CV results

    def logger(self, model, params, mean_cv_score):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_score", mean_cv_score)
            mlflow.sklearn.log_model(model, "rf_model")
        return None


    def grid_search_mlflow_logging(self, model, model_name, param_grid, X, y, save_dir, mlflow_url, cv= 3):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, refit=True)
        
        dagshub.init(repo_owner='Shorya777', repo_name='Insurance_Premium_Prediction', mlflow=True)
        mlflow.set_registry_uri(mlflow_url)

        with mlflow.start_run(run_name= model_name):
            grid_search.fit(X, y)

            self.cv_results = grid_search.cv_results_

            for i, params in enumerate(self.cv_results["params"]):
                mean_cv_score = self.cv_results["mean_test_score"][i]
                std_cv_score = self.cv_results["std_test_score"][i]

                self.logger(model=grid_search.estimator, params=params, mean_cv_score=mean_cv_score)

            best_params = grid_search.best_params_
            mlflow.log_params(best_params)
            best_score = grid_search.best_score_
            mlflow.log_metric("best_mean_cv_score", best_score)
            mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")

            os.makedirs(save_dir, exist_ok=True)
            model_save_path = os.path.join(save_dir, model_name)
            os.makedirs(model_save_path, exist_ok=True)
            print(f"saving {model_name} to {model_save_path}")
            model_path = os.path.join(model_save_path, f"{best_score}_{model_name}.joblib")
            joblib.dump(grid_search.best_estimator_, model_path)
            print("model saved")

            return best_params, best_score

    def models_grid_search_mlflow_logging(self, models, param_grids, X, y, save_dir, mlflow_url, cv=3):
        best_models = {}

        for model_name, model in models.items():
            print(f"Training {model_name}")

            best_param, best_score = self.grid_search_mlflow_logging(model, model_name, param_grids[model_name], X, y, save_dir, cv)
            print(f"Best Parameters for {model_name}: {best_param}")
            print(f"Best Score : {best_score}")
            
            best_models[model_name] = {
                    "best_param": best_param,
                    "best_score": best_score
                    }

        return best_models
