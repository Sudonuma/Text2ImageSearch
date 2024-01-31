from qdrant_client import QdrantClient, models

class Client:
    def __init__(self, host, port, collection_name):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def create_collection(self, vector_size=512):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )

    def upsert_data(self, ids, embeddings, batch_payloads):
        self.client.upsert(
        collection_name=self.collection_name,
        points=models.Batch(
            ids=ids,
            vectors=embeddings,
            payloads=batch_payloads
        )
    )

    def perform_search(self, query_vector, limit=2):
        results = self.client.search(collection_name=self.collection_name,
                      query_vector=query_vector, 
                      limit=limit)
        return results