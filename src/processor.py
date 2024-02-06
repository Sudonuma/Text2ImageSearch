import torch
from transformers import CLIPModel, CLIPProcessor


class Processor:
    """
    Processor class for computing embeddings using CLIP
    (Contrastive Language-Image Pretraining) model.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            device
        )

    def get_embeddings(self, batch: dict) -> dict:
        """
        Get embeddings for a batch of images.

        Args:
            - batch (dict): A dictionary containing batch of image data.

        Returns:
            - dict: A dictionary containing the batch data with embeddings added.

        """
        inputs = self.processor(
            text=None, images=batch["image"], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs).cpu().numpy()

        batch["embeddings"] = outputs
        return batch

    def get_one_embedding(self, text: str) -> torch.Tensor:
        """
        Get embedding for a single text.

        Args:
            - text (str): The text to compute the embedding for.

        Returns:
            - torch.Tensor: The computed embedding tensor.
        """
        inputs = self.processor(text=text, images=None, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs).cpu().numpy()

        return outputs
