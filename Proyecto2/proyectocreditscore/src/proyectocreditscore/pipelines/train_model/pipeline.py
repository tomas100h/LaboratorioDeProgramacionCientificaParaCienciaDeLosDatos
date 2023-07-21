"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_dataset,
                inputs=["dataset_clean", "params:split_params"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="Split_data",
            ),
        ]
    )
