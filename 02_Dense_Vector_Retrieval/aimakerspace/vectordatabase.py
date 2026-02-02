import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Optional
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.metadata = {}  # Store metadata for each text key
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.array, metadata: Optional[Dict] = None) -> None:
        self.vectors[key] = vector
        if metadata is not None:
            self.metadata[key] = metadata

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_metadata: bool = False,
    ) -> List[Tuple]:
        scores = [
            (key, distance_measure(query_vector, vector))
            for key, vector in self.vectors.items()
        ]
        results = sorted(scores, key=lambda x: x[1], reverse=True)[:k]

        if return_metadata:
            # Return (text, score, metadata) tuples
            return [(text, score, self.metadata.get(text, {})) for text, score in results]
        return results

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
        return_metadata: bool = False,
    ) -> List:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure, return_metadata)

        if return_as_text:
            if return_metadata:
                return [(text, metadata) for text, score, metadata in results]
            return [result[0] for result in results]

        return results

    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)

    async def abuild_from_list(self, list_of_text: List[str]) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            self.insert(text, np.array(embedding))
        return self

    async def abuild_from_list_with_metadata(self, chunks_with_metadata: List[Dict]) -> "VectorDatabase":
        """
        Build vector database from chunks that include metadata.

        Args:
            chunks_with_metadata: List of dicts with 'text' and 'metadata' keys

        Returns:
            Self for chaining
        """
        texts = [chunk["text"] for chunk in chunks_with_metadata]
        embeddings = await self.embedding_model.async_get_embeddings(texts)

        for chunk, embedding in zip(chunks_with_metadata, embeddings):
            self.insert(chunk["text"], np.array(embedding), chunk["metadata"])

        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
