"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import get_data, preprocess_companies, preprocess_shuttles, create_model_input_table


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_data,
                inputs=None,
                outputs=['companies', 'shuttles', 'reviews'],
                name="Cargar_datos",
            ),
            node(
                func=preprocess_companies,
                inputs="companies",
                outputs="companies_preprocess",
                name="Preprocess_companies",
            ),
            node(
                func=preprocess_shuttles,
                inputs="shuttles",
                outputs="shuttles_preprocess",
                name="Preprocess_shuttles",
            ),
            node(
                func=create_model_input_table,
                inputs=["shuttles_preprocess", "companies_preprocess", "reviews"],
                outputs="reviews_preprocess",
                name="Preprocess_reviews",
            )
        ]
    )
