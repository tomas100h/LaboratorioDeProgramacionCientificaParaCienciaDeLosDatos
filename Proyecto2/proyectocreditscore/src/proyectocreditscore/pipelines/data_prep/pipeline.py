"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_and_clean_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_and_clean_dataset,
                inputs=["dataset_raw"],
                outputs="dataset_clean",
                name="Load_and_clean_data",
            ),
        ]
    )
