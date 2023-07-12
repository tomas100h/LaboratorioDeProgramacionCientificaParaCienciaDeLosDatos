"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""

import logging
from datetime import datetime
from typing import Dict

import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from xgboost import XGBRegressor


def split_data(data: pd.DataFrame, params: Dict):

    shuffled_data = data.sample(frac=1, random_state=params["random_state"])
    rows = shuffled_data.shape[0]

    train_ratio = params["train_ratio"]
    valid_ratio = params["valid_ratio"]

    train_idx = int(rows * train_ratio)
    valid_idx = train_idx + int(rows * valid_ratio)

    assert rows > valid_idx, "test split should not be empty"

    target = params["target"]
    X = shuffled_data.drop(columns=target)
    y = shuffled_data[[target]]

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_valid, y_valid = X[train_idx:valid_idx], y[train_idx:valid_idx]
    X_test, y_test = X[valid_idx:], y[valid_idx:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_mae")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model


def train_model(X_train, y_train, X_val, y_val):
    now = datetime.now()
    experimentname = "Experiment" + now.strftime("%Y-%m-%d_%H-%M")
    experiment_id = mlflow.create_experiment(experimentname)

    with mlflow.start_run(run_name="Linear Regression Default"):
        linear_reg = LinearRegression()
        linear_reg.fit(X_train, y_train)
        y_pred_linear = linear_reg.predict(X_val)
        mae_linear = mean_absolute_error(y_val, y_pred_linear)
        mlflow.log_metric("valid_mae", mae_linear)

    with mlflow.start_run(run_name="Random Forest Regression Default"):
        random_forest_reg = RandomForestRegressor()
        random_forest_reg.fit(X_train, y_train)
        y_pred_rf = random_forest_reg.predict(X_val)
        mae_rf = mean_absolute_error(y_val, y_pred_rf)
        mlflow.log_metric("valid_mae", mae_rf)

    with mlflow.start_run(run_name="SVR Default"):
        svr_reg = SVR()
        svr_reg.fit(X_train, y_train)
        y_pred_svr = svr_reg.predict(X_val)
        mae_svr = mean_absolute_error(y_val, y_pred_svr)
        mlflow.log_metric("valid_mae", mae_svr)

    with mlflow.start_run(run_name="XGBoost Regression Default"):
        xgb_reg = XGBRegressor()
        xgb_reg.fit(X_train, y_train)
        y_pred_xgb = xgb_reg.predict(X_val)
        mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
        mlflow.log_metric("valid_mae", mae_xgb)

    with mlflow.start_run(run_name="LGBM Regression Default"):
        lgbm_reg = LGBMRegressor()
        lgbm_reg.fit(X_train, y_train)
        y_pred_lgbm = lgbm_reg.predict(X_val)
        mae_lgbm = mean_absolute_error(y_val, y_pred_lgbm)
        mlflow.log_metric("valid_mae", mae_lgbm)

    return get_best_model(experiment_id)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info(f"Model has a Mean Absolute Error of {mae} on test data.")
