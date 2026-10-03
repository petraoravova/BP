# author: Petra Oravová
# file: infer.py
# description: Functions for performing cow identification and verification using trained embedding models.
#              Includes muzzle detection, embedding extraction, and database comparison.

import torch
from PIL import Image
from db.utils import identify_cow
from identification.model import MODEL_REGISTRY
from detection.detect_and_crop import detect_and_crop
import cv2
import math


def infer(image_path, model_name="mobilenet", model_path="models/mobilenet.pth", threshold=0.5):
    """
    Identifies a cow in the given image by detecting the muzzle, extracting an embedding,
    and comparing it with the database.

    Args:
        image_path (str): Path to the input image.
        model_name (str, optional): Model architecture name. Defaults to 'mobilenet'.
        model_path (str, optional): Path to the trained model weights. Defaults to 'models/mobilenet.pth'.
        threshold (float, optional): Distance threshold for identification match. Defaults to 0.5.

    Returns:
        Tuple[Tuple, Tuple]: A tuple containing the matched cow ID and distance, or a message if detection fails,
                                and bounding box (tuple of integers representing coordinates).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_class = MODEL_REGISTRY[model_name]
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Detect the cow's muzzle and crop the image
    cropped_image, bbox = detect_and_crop(image_path)

    if cropped_image is None:
        print("No cow detected")
        return "No cow detected", None

    cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    # Identify cow from embedding
    embedding = model.get_embedding(cropped_image_pil, device)
    result = identify_cow(embedding, threshold=threshold)

    print(result)
    return result, bbox


def verify(image1_path, image2_path, model_name="mobilenet", model_path="models/mobilenet.pth", threshold=0.5):
    """
    Verifies whether two input images belong to the same cow based on embedding similarity.

    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.
        model_name (str, optional): Model architecture name. Defaults to 'mobilenet'.
        model_path (str, optional): Path to the trained model weights. Default to 'models/mobilenet.pth'.
        threshold (float, optional): Distance threshold for verification. Defaults to 0.5.

    Returns:
        Tuple[bool, float, Tuple, Tuple]: A tuple containing a boolean (True/False), distance (float), and two bounding
                                            boxes (each as a tuple of integers representing coordinates).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class = MODEL_REGISTRY[model_name]
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Detect muzzles and crop the images
    crop1, bbox1 = detect_and_crop(image1_path)
    crop2, bbox2 = detect_and_crop(image2_path)
    if crop1 is None or crop2 is None:
        print("No cow detected in one or both images.")
        return "No cow detected in one or both images.", None, None, None

    pil1 = Image.fromarray(cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB))
    pil2 = Image.fromarray(cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB))

    # Get embeddings
    emb1 = model.get_embedding(pil1, device)
    emb2 = model.get_embedding(pil2, device)

    # Compute Euclidean distance between embeddings
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(emb1, emb2)))

    same = distance < threshold

    return same, distance, bbox1, bbox2
