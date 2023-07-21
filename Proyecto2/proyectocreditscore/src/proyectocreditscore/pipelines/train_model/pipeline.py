"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_dataset, scale_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_dataset,
                inputs=["dataset_clean", "params:split_params"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="Split_data",
            ),
            node(
                func=scale_features,
                inputs=["X_train", "X_test"],
                outputs=["X_train_preprocessed", "X_test_preprocessed"],
                name="Scale_features",
            ),
        ]
    )
