import torch
from transformers import CLIPProcessor, CLIPModel

class Processor:
    def __init__(self, device):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    def get_embeddings(self, batch):
        inputs = self.processor(text=None, images=batch['image'], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs).cpu().numpy()
        
        batch['embeddings'] = outputs
        return batch
    
    def get_one_embedding(self, text):
        inputs = self.processor(text=text, images=None, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs).cpu().numpy()
        
        return outputs


    