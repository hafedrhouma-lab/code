from ultron.runners.text_embeddings import TextEmbeddingsModelName

TABLE_TO_EMBEDDINGS_MODEL: dict[str, TextEmbeddingsModelName] = {
    "all_items_embeddings_metadata": TextEmbeddingsModelName.ALL_MINILM_L6_V2
}


def validate_embedding_format(table_name: str, model_name: TextEmbeddingsModelName):
    assert table_name in TABLE_TO_EMBEDDINGS_MODEL, f"unknown table name `{table_name}`"
    assert TABLE_TO_EMBEDDINGS_MODEL[table_name] == model_name, "incompatible embeddings"
