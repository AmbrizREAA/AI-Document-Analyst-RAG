"""HuggingFace embedding model factory."""

from langchain_huggingface import HuggingFaceEmbeddings

from platform_layer.config.settings import EMBEDDING_MODEL_NAME


def get_embeddings():
    """Create the HuggingFace embeddings model used across the app."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
