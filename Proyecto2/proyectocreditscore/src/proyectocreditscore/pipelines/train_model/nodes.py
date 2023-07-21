"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""

from datetime import datetime
from typing import Dict, Tuple

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def split_dataset(
    df: pd.DataFrame, params: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training and testing sets.

    Parameters:
    df (pd.DataFrame): The input DataFrame to split.
    params (Dict): A dictionary containing parameters for splitting

    Returns:
    list: A list containing four DataFrames: X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=["credit_score"])
    y = df[["credit_score"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_ratio"], random_state=params["seed"]
    )
    return X_train, X_test, y_train, y_test


def column_transformation() -> ColumnTransformer:
    """
    This function creates a column transformer for pre-processing of features
    before applying a machine learning model.

    The function includes separate transformations for numeric, categorical,
    and binary features. The numeric features are scaled
    using a RobustScaler to minimize the effect of outliers. Categorical
    features are encoded using a OneHotEncoder to convert them
    into numeric values, which are more suitable for machine learning models.
    Binary features are passed through without any
    transformation.

    Please note, the input and output dataframes are in pandas DataFrame
    format.

    Numeric Features are:
        - "age",
        - "annual_income",
        - "monthly_inhand_salary",
        - "num_bank_accounts",
        - "num_credit_card",
        - "interest_rate",
        - "num_of_loan",
        - "delay_from_due_date",
        - "num_of_delayed_payment",
        - "changed_credit_limit",
        - "num_credit_inquiries",
        - "outstanding_debt",
        - "credit_utilization_ratio",
        - "credit_history_age",
        - "total_emi_per_month",
        - "amount_invested_monthly",
        - "monthly_balance",

    Categorical Features are:
        - "occupation",
        - "payment_behaviour"

    Binary Features are:
        - "payment_of_min_amount"

    Returns:
    ColumnTransformer: The constructed ColumnTransformer instance that could
    be further integrated into a machine learning pipeline.
    """
    numeric_features = [
        "age",
        "annual_income",
        "monthly_inhand_salary",
        "num_bank_accounts",
        "num_credit_card",
        "interest_rate",
        "num_of_loan",
        "delay_from_due_date",
        "num_of_delayed_payment",
        "changed_credit_limit",
        "num_credit_inquiries",
        "outstanding_debt",
        "credit_utilization_ratio",
        "credit_history_age",
        "total_emi_per_month",
        "amount_invested_monthly",
        "monthly_balance",
    ]

    categorical_features = ["occupation", "payment_behaviour", "payment_of_min_amount"]

    binary_features = []

    numeric_transformer = RobustScaler()

    categorical_transformer = Pipeline(
        steps=[("encode", OneHotEncoder(sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", "passthrough", binary_features),
        ],
        remainder="drop",
    ).set_output(transform="pandas")

    return preprocessor


def select_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> pd.DataFrame:
    # Preprocesar las características utilizando el preprocessor
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    y_train = y_train.values.flatten()
    y_test = y_test.values.flatten()

    # Crear una lista para almacenar los resultados de las métricas
    metric_results = []

    # Definir los clasificadores
    classifiers = {
        "Dummy": DummyClassifier(strategy="stratified"),
        "Logistic Regression": LogisticRegression(),
        # "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "LightGBM": LGBMClassifier(),
        "XGBoost": XGBClassifier(),
    }

    # Iniciar la sesión de MLflow
    now = datetime.now()
    experimentname = "Experiment" + now.strftime("%Y-%m-%d_%H-%M")
    experiment_id = mlflow.create_experiment(experimentname)

    # Registrar el preprocesador con MLflow
    with mlflow.start_run(run_name="Preprocessor"):
        mlflow.sklearn.log_model(preprocessor, "preprocessor")

    # Iterar sobre los clasificadores
    for clf_name, clf in classifiers.items():
        with mlflow.start_run(run_name=clf_name):
            # Entrenar el clasificador con los conjuntos preprocesados
            clf.fit(X_train_preprocessed, y_train)
            print("Entrenando", clf_name)
            # Realizar predicciones en el conjunto de prueba
            y_pred = clf.predict(X_test_preprocessed)
            print("Entrenado", clf_name)

            # Calcular las métricas y guardar los resultados
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            metric_results.append(
                {
                    "Model": clf_name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-score": f1,
                }
            )

            # Imprimir el reporte de clasificación para cada clasificador
            print(f"Classification Report for {clf_name}:")
            print(classification_report(y_test, y_pred))
            print("-" * 80)

            # Registrar las métricas en MLflow
            mlflow.log_metrics(
                {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-score": f1,
                }
            )

            # Guardar el modelo en MLflow
            mlflow.sklearn.log_model(clf, "model")

    # Convertir la lista de resultados de métricas a un DataFrame
    metrics_df = pd.DataFrame(metric_results)

    # Ordenar el DataFrame según los valores de la métrica seleccionada (por ejemplo, Accuracy)
    selected_metric = "Accuracy"
    sorted_metrics_df = metrics_df.sort_values(by=selected_metric, ascending=False)

    print("Results sorted by", selected_metric)
    print(sorted_metrics_df)
    return sorted_metrics_df
