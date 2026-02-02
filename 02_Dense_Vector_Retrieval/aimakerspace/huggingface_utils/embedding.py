from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class HuggingFaceEmbeddingModel:
    """
    HuggingFace embedding model using sentence-transformers.
    Provides the same interface as the OpenAI EmbeddingModel for compatibility.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the HuggingFace embedding model.

        :param model_name: Name of the sentence-transformers model to use

        Popular models:
        - sentence-transformers/all-MiniLM-L6-v2 (384 dim) - Fast and efficient
        - sentence-transformers/all-mpnet-base-v2 (768 dim) - Higher quality
        - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dim) - Multilingual
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

        print(f"Loaded {model_name}")
        print(f"Max sequence length: {self.model.max_seq_length}")
        print(f"Embedding dimension: {self.embedding_dimension}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.

        :param text: Text to embed
        :return: Embedding as a list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def get_embeddings(self, list_of_text: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts (batch processing).

        :param list_of_text: List of texts to embed
        :return: List of embeddings
        """
        embeddings = self.model.encode(list_of_text, convert_to_numpy=True, show_progress_bar=True)
        return [embedding.tolist() for embedding in embeddings]

    async def async_get_embedding(self, text: str) -> List[float]:
        """
        Async version of get_embedding for compatibility.
        Note: sentence-transformers doesn't have native async support,
        so this just calls the sync version.

        :param text: Text to embed
        :return: Embedding as a list of floats
        """
        return self.get_embedding(text)

    async def async_get_embeddings(self, list_of_text: List[str]) -> List[List[float]]:
        """
        Async version of get_embeddings for compatibility.
        Note: sentence-transformers doesn't have native async support,
        so this just calls the sync version.

        :param list_of_text: List of texts to embed
        :return: List of embeddings
        """
        return self.get_embeddings(list_of_text)


if __name__ == "__main__":
    import asyncio

    # Test the embedding model
    embedding_model = HuggingFaceEmbeddingModel()

    # Test single embedding
    print("\nTesting single embedding:")
    embedding = embedding_model.get_embedding("Hello, world!")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")

    # Test batch embeddings
    print("\nTesting batch embeddings:")
    embeddings = embedding_model.get_embeddings(["Hello, world!", "Goodbye, world!"])
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Each embedding dimension: {len(embeddings[0])}")

    # Test async (for compatibility)
    print("\nTesting async embedding:")
    async_embedding = asyncio.run(embedding_model.async_get_embedding("Async test"))
    print(f"Async embedding dimension: {len(async_embedding)}")
