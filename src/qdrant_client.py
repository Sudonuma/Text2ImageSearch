import os
from typing import List

from qdrant_client import QdrantClient, models


class Client:
    """
    Client class for interacting with Qdrant for vector indexing and searching.
    """

    def __init__(self, collection_name: str, port: int = 6333, cloud: bool = True):

        """
        Initialize the Qdrant client.

        Args:
            - host (str): The host address of the Qdrant server.
            - port (int): The port number of the Qdrant server.
            - collection_name (str): The name of the collection in Qdrant.
            - cloud (bool): True if you want to use qdrant cloud, False if you want to use localhost.
                            Note(run docker if you are using localhost)
        """
        
        if cloud:
            self.client = QdrantClient(
                url=os.environ.get("API_URL"),
                api_key=os.environ.get("API_KEY"),
            )
        else:
            self.client = QdrantClient(host="localhost", port=port)
        self.collection_name = collection_name

    def create_collection(self, vector_size: int = 512) -> None:
        """
        Create a collection in Qdrant.

        Args:
            - vector_size (int): The dimensions each vector corresponding to the image in the embedding.
                                 CLIP output is 512.
        """
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )

    def upsert_data(self, ids: List, embeddings: List, batch_payloads: List):
        """
        Update or insert data (IDs, embeddings, and payloads) into the collection.

        Args:
            - ids (List): List of IDs for the data points.
            - embeddings (List): List of embedding vectors for the data points.
            - batch_payloads (List): List of payloads associated with the data points.
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(ids=ids, vectors=embeddings, payloads=batch_payloads),
        )

    def perform_search(self, query_vector: List[float], limit: int = 2):
        """
        Perform a search in the collection.

        Args:
            - query_vector (List[float]): The query vector for similarity search.
            - limit (int): The maximum number of results to return. Defaults to 2.

        Returns:
        # TODO check the result output type.
        """
        results = self.client.search(
            collection_name=self.collection_name, query_vector=query_vector, limit=limit
        )
        return results
