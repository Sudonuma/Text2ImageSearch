from datasets import load_dataset
from src.processor import Processor
from src.qdrant_client import Client
import numpy as np
import torch
import os
import PIL

# TODO add logger

def see_images(results, dataset, top_k=2):
    for i in range(top_k):

        result_id = results[i].id
        score    = results[i].score
        image = dataset['image'][result_id]

        print(f"This image score was {score}")
        image.save(str(i)+'test.jpg')
        
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset("imagefolder", data_dir="./dataset/", split="train")
    processor = Processor(device)
    
    client = Client(host="localhost", port=6333, collection_name="Flickr8k")
    client.create_collection()

    dataset = dataset.map(processor.get_embeddings, batched=True, batch_size=16)
    
    embeddings = np.array(dataset['embeddings'])
    np.save("vectors", embeddings, allow_pickle=False)

    payload = dataset.select_columns(['image']).to_pandas().to_dict(orient="records")
    
    for element in payload:
        filename = os.path.basename(element['image']['path'])
        image_id = os.path.splitext(filename)[0]
        element['image']['image_id'] = image_id
        

    ids = list(range(dataset.num_rows))
    batch_size = 16

    for i in range(0, dataset.num_rows, batch_size):

        low_idx = min(i+batch_size, dataset.num_rows)

        batch_of_ids = ids[i: low_idx]
        batch_of_embs = embeddings[i: low_idx]
        #  save image ID (filename) instead of path in payload
        batch_of_payloads = payload[i: low_idx]
        # print(batch_of_payloads)

        # also make sure everything in payload can change to json
        # Ensure embeddings are lists
        batch_of_embs = [emb if isinstance(emb, list) else emb.tolist() for emb in batch_of_embs]

        client.upsert_data(batch_of_ids, batch_of_embs, batch_of_payloads)


    # search: change to another script
    search_text = "two dogs playing"

    one_embed = processor.get_one_embedding(search_text)
    result = client.perform_search(one_embed[0])
    print(result)

    see_images(result, dataset, top_k=2)

if __name__ == "__main__":
    main()