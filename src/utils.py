import os
from typing import List, Tuple

import numpy as np

# TODO add logger


def get_results(results, dataset, top_k: int = 2) -> Tuple[List, List]:
    """
    Extracts scores from the search results and the corresponding image from the dataset using the image ID.

    Args:
        - results : search results.
        - dataset : Dataset containing the images.
        - top_k (int): Number of top results to extract. Defaults to 2.

    Returns:
        - Tuple[List, List]: List of images and corresponding scores.
    """

    images = []
    scores = []
    # image_rel_path = []
    for i in range(top_k):
        result_id = results[i].id
        score = results[i].score
        image = dataset["image"][result_id]
        images.append(image)
        scores.append(score)

    return images, scores


def embed_data(dataset, processor, client) -> None:
    """
    Embeds the dataset data and upserts it into the Qdrant collection.

    Args:
        - dataset : The dataset to embed and upsert.
        - processor (Processor): Processor instance for embedding.
        - client (Client): Client instance for interacting with Qdrant.
    """
    dataset = dataset.map(processor.get_embeddings, batched=True, batch_size=16)
    embeddings = np.array(dataset["embeddings"])

    np.save("vectors", embeddings, allow_pickle=False)
    payload = dataset.select_columns(["image"]).to_pandas().to_dict(orient="records")

    for element in payload:
        filename = os.path.basename(element["image"]["path"])
        image_id = os.path.splitext(filename)[0]
        element["image"]["image_id"] = image_id

    ids = list(range(dataset.num_rows))
    batch_size = 16
    print("start embedding")
    # add tqdm
    for i in range(0, dataset.num_rows, batch_size):

        low_idx = min(i + batch_size, dataset.num_rows)

        batch_of_ids = ids[i:low_idx]
        batch_of_embs = embeddings[i:low_idx]
        batch_of_payloads = payload[i:low_idx]
        batch_of_embs = [
            emb if isinstance(emb, list) else emb.tolist() for emb in batch_of_embs
        ]

        client.upsert_data(batch_of_ids, batch_of_embs, batch_of_payloads)

    print("end embedding")


def search(search_text: str, processor, client, dataset) -> Tuple[List, List]:
    """
    Performs a search in the Qdrant collection based on the search text.

    Args:
        - search_text (str): The text to search for.
        - processor (Processor): Processor instance for embedding.
        - client (Client): Client instance for interacting with Qdrant.
        - dataset: Dataset containing the images.

    Returns:
        - Tuple[List, List]: List of images and corresponding scores.
    """
    one_embed = processor.get_one_embedding(search_text)
    result = client.perform_search(one_embed[0])
    images, scores = get_results(result, dataset, top_k=2)
    return images, scores
