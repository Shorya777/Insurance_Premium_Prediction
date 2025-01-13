from box.exceptions import BoxValueError
from box import ConfigBox
from pathlib import Path 
from ensure import ensure_annotations
from logger_config import get_logger
import yaml
import pandas as pd
import gc
import joblib 
import numpy as np
from itertools import product

logger = get_logger(__name__)

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

def custom_grid_search(model_name, model, save_dir, param_grid, X, y, cv=2, scoring='mean_squared_error'):
    gc.collect()
    if cv < 2:
        raise ValueError("Cross-validation requires at least 2 folds (cv >= 2).")

    best_score = np.inf
    best_params = None

    param_combinations = list(product(*param_grid.values()))
    param_keys = list(param_grid.keys())

    for i, combination in enumerate(param_combinations, 1):
        params = dict(zip(param_keys, combination))
        model.set_params(**params)
        print(f"Training model {i}/{len(param_combinations)} with params: {params}")

        np.random.seed(42)
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        fold_sizes = np.full(cv, len(X) // cv)
        fold_sizes[:len(X) % cv] += 1
        folds = np.split(indices, np.cumsum(fold_sizes)[:-1])

        cv_scores = []

        for fold_idx, valid_indices in enumerate(folds):
            train_indices = np.concatenate([folds[i] for i in range(cv) if i != fold_idx])
            X_train, X_valid = X[train_indices], X[valid_indices]
            y_train, y_valid = y[train_indices], y[valid_indices]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_valid)

            if scoring == 'mean_squared_error':
                score = np.mean((y_valid - y_pred) ** 2)
            else:
                raise ValueError(f"Scoring method {scoring} is not supported")

            cv_scores.append(score)

        mean_cv_score = np.mean(cv_scores)

        if mean_cv_score < best_score:
            best_score = mean_cv_score
            best_params = params

            os.makedirs(save_dir, exist_ok=True)
            model_save_path = os.path.join(save_dir, model_name)
            os.makedirs(model_save_path, exist_ok=True)
            print(f"saving {model_name} to {model_save_path}")
            model_path = os.path.join(model_save_path, f"{best_score}_{model_name}.joblib")
            joblib.dump(model, model_path)
            print("model saved")

    return best_params, best_score

def custom_grid_search_models(models, save_dir, param_grid, X_train, y_train, mlflow_url, cv=2):
    best_models = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        best_params, best_score = custom_grid_search(model_name, model, save_dir, param_grid[model_name], X_train, y_train, mlflow_url,  cv=cv)

        print(f"Best Parameters for {model_name}: {best_params}")
        print(f"Best Score (MSE): {best_score}")

        best_models[model_name] = {
            'best_params': best_params,
            'best_score': best_score,
        }

    return best_models
