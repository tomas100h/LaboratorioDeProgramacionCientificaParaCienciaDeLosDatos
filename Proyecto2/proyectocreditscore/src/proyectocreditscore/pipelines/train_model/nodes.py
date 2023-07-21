"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""

import pandas as pd
from typing import Dict, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


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


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scales the features of the dataset.

    The function uses a RobustScaler for numeric features, OneHotEncoder for
    categorical features, and passes through binary features.

    Parameters:
    X_train (pd.DataFrame): The training data to be scaled.
    X_test (pd.DataFrame): The testing data to be scaled.

    Returns:
    tuple: A tuple containing the scaled versions of X_train and X_test.
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

    categorical_features = ["occupation", "payment_behaviour"]

    binary_features = ["payment_of_min_amount"]

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
        remainder="passthrough",
    ).set_output(transform="pandas")

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed
