# author: Petra Oravová
# file: model.py
# description: This module defines a base class for extracting image embeddings using deep CNNs,
#              along with specific subclasses that implement EfficientNet, ResNet, and MobileNet architectures.
# used documentation and tutorials:
# - https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648
# - https://pytorch.org/vision/main/models.html

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn


class EmbeddingModel(nn.Module):
    """
    Base class for embedding models.
    Defines a common interface for extracting L2-normalized embeddings from images.
    """
    def __init__(self):
        super(EmbeddingModel, self).__init__()

    def forward(self, x):
        """
        Forward pass through the model.

        Passes the input image tensor through the base feature extractor
        and the projection layer to obtain the final embedding.

        Args:
            x (Tensor): Input tensor representing the image.

        Returns:
            Tensor: 256-dimensional embedding tensor.
        """
        x = self.base(x)
        x = self.embedding_layer(x)
        return x

    def get_embedding(self, image_path, device):
        """
        Loads an image, preprocesses it, and returns its embedding.

        Args:
            image_path (str or PIL.Image): Path to the image file or a PIL.Image.
            device (torch.device): Torch device (e.g., "cuda" or "cpu").

        Returns:
            list: L2-normalized embedding as a list.
        """
        if isinstance(image_path, str):
            pil_img = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Image.Image):
            pil_img = image_path
        else:
            raise TypeError(f"Unsupported type {type(image_path)} for get_embedding")

        image = self.transform(pil_img).unsqueeze(0).to(device)
        self.eval()
        with torch.no_grad():
            embedding = self.forward(image)
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.squeeze(0).cpu().numpy().tolist()


class EfficientNetB1Embedding(EmbeddingModel):
    """
    EfficientNet-B1 based embedding model.
    Removes the classifier and adds a 1280→256 projection layer.
    """
    def __init__(self):
        super(EfficientNetB1Embedding, self).__init__()
        self.base = models.efficientnet_b1(weights='DEFAULT')
        self.base.classifier = nn.Identity()
        self.embedding_layer = nn.Linear(1280, 256)
        self.transform = models.EfficientNet_B1_Weights.IMAGENET1K_V2.transforms()


class ResNetEmbedding(EmbeddingModel):
    """
    ResNet-50 based embedding model.
    Replaces the FC layer with a 2048→256 projection layer.
    """
    def __init__(self):
        super(ResNetEmbedding, self).__init__()
        self.base = models.resnet50(weights='DEFAULT')
        self.base.fc = nn.Identity()
        self.embedding_layer = nn.Linear(2048, 256)
        self.transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()


class MobileNetEmbedding(EmbeddingModel):
    """
    MobileNet_V3_Large based embedding model.
    Removes classifier and adds a 960→256 projection layer.
    """
    def __init__(self):
        super(MobileNetEmbedding, self).__init__()
        self.base = models.mobilenet_v3_large(weights='DEFAULT')
        self.base.classifier = nn.Identity()
        self.embedding_layer = nn.Linear(960, 256)
        self.transform = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms()


# Dictionary to access available models by name
MODEL_REGISTRY = {
    "efficientnet": EfficientNetB1Embedding,
    "resnet": ResNetEmbedding,
    "mobilenet": MobileNetEmbedding,
}
