"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""

from typing import List, Dict
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(df: pd.DataFrame,  params: Dict) -> List(pd.DataFrame):
    """Preprocesses the data"""
    X = df.drop(columns=['credit_score'])
    y = df['credit_score']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['test_ratio'], random_state=params['seed']
        )
    return X_train, X_test, y_train, y_test
