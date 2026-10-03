# author: Petra Oravová
# file: triplet_dataset.py
# description: Contains a custom PyTorch Dataset class for training and evaluation of models using triplet loss.
#              It dynamically generates triplets (anchor, positive, negative) for training
#              or precomputes them for evaluation.
# used documentation and tutorials:
# - https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import os
import random
from glob import glob
from PIL import Image
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    """
    PyTorch Dataset for generating image triplets (anchor, positive, negative) used in metric learning.
    Supports both dynamic generation for training and static generation for evaluation.
    """
    def __init__(self, root_dir, transform=None, is_training=True):
        """
        Initializes the dataset and optionally precompute triplets.

        Args:
            root_dir (str): Directory containing subfolders for each class with images.
            transform (Callable, optional): Transformations to apply to images.
            is_training (bool): If True, generates triplets on-the-fly. If False, precomputes them.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training

        # Create a mapping from class name to list of image paths
        self.class_to_images = {
            class_name: glob(os.path.join(root_dir, class_name, '*'))
            for class_name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, class_name))
        }

        # Keep only classes that have more than one image
        self.classes = [cls for cls, imgs in self.class_to_images.items() if len(imgs) > 1]

        # Precompute triplets if not in training mode
        if not is_training:
            self.triplets = self.create_triplets()
        else:
            self.triplets = None

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of triplets (if testing) or total number of images (if training).
        """
        if self.is_training:
            return sum(len(imgs) for imgs in self.class_to_images.values())
        else:
            return len(self.triplets)

    def load_images(self, anchor_path, positive_path, negative_path):
        """
        Loads and transforms triplet images from given file paths.

        Args:
            anchor_path (str): Path to the anchor image.
            positive_path (str): Path to the positive image (same class as anchor).
            negative_path (str): Path to the negative image (different class).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Transformed anchor, positive, and negative images.
        """
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def create_triplets(self):
        """
        Precomputes all triplets for evaluation. Ensures reproducibility during testing.

        Returns:
            List[Tuple[str, str, str]]: List of (anchor, positive, negative) image triplets.
        """
        triplets = []
        for anchor_class in self.classes:
            anchor_images = self.class_to_images[anchor_class]
            for anchor_img in anchor_images:
                positive_img = random.choice([img for img in anchor_images if img != anchor_img])
                negative_class = random.choice([cls for cls in self.classes if cls != anchor_class])
                negative_img = random.choice(self.class_to_images[negative_class])

                triplets.append(self.load_images(anchor_img, positive_img, negative_img))
        return triplets

    def create_triplet(self):
        """
        Randomly generates a new triplet for training.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Anchor, positive, and negative images.
        """
        anchor_class = random.choice(self.classes)
        anchor_imgs = self.class_to_images[anchor_class]

        anchor_path, positive_path = random.sample(anchor_imgs, 2)

        negative_class = random.choice([cls for cls in self.classes if cls != anchor_class])
        negative_path = random.choice(self.class_to_images[negative_class])

        return self.load_images(anchor_path, positive_path, negative_path)

    def __getitem__(self, idx):
        """
        Gets one triplet either from precomputed set or generate new one.

        Args:
            idx (int): Index of the triplet (used only if not training).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Anchor, positive, and negative transformed images.
        """
        if self.is_training:
            return self.create_triplet()
        else:
            return self.triplets[idx]
