"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["data", "params:split_params"],
                outputs=["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"],
                name="Split_data",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "X_val", "y_val"],
                outputs="best_model",
                name="Train_data",
            ),
            node(
                func=evaluate_model,
                inputs=["best_model", "X_test", "y_test"],
                outputs=None,
                name="Test_model",
            ),
        ]
    )
