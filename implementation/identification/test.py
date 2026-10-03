# author: Petra Oravová
# file: test.py
# description:  This script provides functions for evaluating models for cow identification.
#               It includes tools for pair generation, metric computation, and visualization of embedding distances.
# used documentation and tutorials:
# - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
# - https://matplotlib.org/stable/api/index.html
# - https://docs.python.org/3/library/itertools.html#itertools.combinations

import os
import math
import torch
from identification.model import MODEL_REGISTRY
from db.utils import identify_cow
from db.utils import cow_exists
import random
import itertools
import matplotlib.pyplot as plt
import json


def plot_genuine_impostor_histogram(distances, model, save_path=None):
    """
    Plots and optionally saves a histogram of genuine vs. impostor pair distances.

    Args:
        distances (dict): Dictionary with lists of 'genuine' and 'impostor' distances.
        model (str): Model name used for the plot title.
        save_path (str, optional): Path to save the histogram as a PDF. Defaults to None.
    """
    model_names = {
        "efficientnet": "EfficientNet_B1",
        "resnet": "ResNet-50",
        "mobilenet": "MobileNet_V3_Large"
    }
    genuine_distances = distances['genuine']
    impostor_distances = distances['impostor']
    plt.figure(figsize=(8, 5))
    plt.hist(genuine_distances, bins=30, alpha=0.5, label='Shodné páry ($\t{{genuine}}$)', color='green', density=True)
    plt.hist(impostor_distances, bins=30, alpha=0.5, label='Rozdílné páry ($\t{{impostor}}$)', color='red',
             density=True)
    plt.xlabel('Eukleidovská vzdálenost', labelpad=8)
    plt.ylabel('Počet případů (normalizováno)', labelpad=8)
    plt.title(f'Histogram vzdáleností ({model_names[model]}): $\t{{genuine}}$ $\t{{vs}}$ $\t{{impostor}}$', pad=12)
    plt.legend()
    plt.grid(True)

    plt.xlim(left=0)

    plt.ylim(bottom=0)

    if save_path:
        plt.savefig(save_path, format="pdf")
    plt.show()


def generate_and_save_pairs(save_path="data/pairs.json", max_same_pairs_per_cow=15):
    """
    Generates image pairs (genuine and impostor) from the newcows_test dataset and save them to a JSON file.

    Args:
        save_path (str, optional): Output path for saving generated pairs. Defaults to "data/pairs.json".
        max_same_pairs_per_cow (int, optional): Maximum number of same-cow image pairs per cow. Defaults to 15.
    """
    test_folder = 'data/id_dataset_split/newcows_test'
    cow_folders = [f for f in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, f))]
    image_paths = {
        cow: [os.path.join(test_folder, cow, img) for img in os.listdir(os.path.join(test_folder, cow))
              if img.lower().endswith(('.jpg', '.png'))]
        for cow in cow_folders
    }

    # Genuine pairs
    same_pairs = []
    for cow, imgs in image_paths.items():
        if len(imgs) >= 2:
            combos = list(itertools.combinations(imgs, 2))
            random.shuffle(combos)
            same_pairs.extend(combos[:min(max_same_pairs_per_cow, len(combos))])

    # Impostor pairs
    different_pairs = []
    different_combinations = list(itertools.combinations(cow_folders, 2))
    random.shuffle(different_combinations)
    for cow1, cow2 in different_combinations[:len(same_pairs)]:
        img1 = random.choice(image_paths[cow1])
        img2 = random.choice(image_paths[cow2])
        different_pairs.append((img1, img2))

    pairs = [(img1, img2, True) for img1, img2 in same_pairs] + [(img1, img2, False) for img1, img2 in
                                                                 different_pairs]
    random.shuffle(pairs)

    with open(save_path, "w") as f:
        json.dump(pairs, f)

    print(f"Saved {len(pairs)} pairs to {save_path}")


def evaluate_model_on_dataset(model, test_folder, threshold, mode="identification"):
    """
    Evaluates a model on the test dataset using either identification or new_cows (verification) mode.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        test_folder (str): Path to the dataset.
        threshold (float): Distance threshold for classification.
        mode (str, optional): Evaluation mode - 'identification' or 'new_cows'. Defaults to 'identification'.

    Returns:
        dict: Dictionary of evaluation statistics.
    """
    stats = {
        "genuine_distances": [],
        "impostor_distances": [],
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
        "max_correct_distance": 0.0,
        "min_incorrect_distance": float('inf'),
        "total": 0,
        "tp_distance_sum": 0.0,
        "fp_distance_sum": 0.0,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    if mode == "identification":
        for cow_id in os.listdir(test_folder):
            cow_path = os.path.join(test_folder, cow_id)
            if not os.path.isdir(cow_path):
                continue

            for file in os.listdir(cow_path):
                if not file.lower().endswith(('.jpg', '.png')):
                    continue

                file_path = os.path.join(cow_path, file)
                embedding = model.get_embedding(file_path, device)
                in_db = cow_exists(cow_id)
                result = identify_cow(embedding, threshold=threshold)

                stats["total"] += 1

                if result[0] is None:
                    if not in_db:
                        stats["true_negative"] += 1
                    else:
                        stats["false_negative"] += 1
                    continue

                predicted_cow_id, distance = result
                if predicted_cow_id == cow_id:
                    stats["genuine_distances"].append(distance)
                    stats["true_positive"] += 1
                    stats["max_correct_distance"] = max(stats["max_correct_distance"], distance)
                    stats["tp_distance_sum"] += distance
                else:
                    stats["impostor_distances"].append(distance)
                    stats["false_positive"] += 1
                    stats["min_incorrect_distance"] = min(stats["min_incorrect_distance"], distance)
                    stats["fp_distance_sum"] += distance
    elif mode == "new_cows":
        with open("data/pairs.json", "r") as f:
            pairs = json.load(f)

        for img_path1, img_path2, is_same_cow in pairs:
            emb1 = model.get_embedding(img_path1, device)
            emb2 = model.get_embedding(img_path2, device)
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(emb1, emb2)))
            stats["total"] += 1
            if is_same_cow:
                stats["genuine_distances"].append(distance)
            else:
                stats["impostor_distances"].append(distance)
            if threshold < 0 or distance <= threshold:
                if is_same_cow:
                    stats["true_positive"] += 1
                    stats["max_correct_distance"] = max(stats["max_correct_distance"], distance)
                    stats["tp_distance_sum"] += distance
                else:
                    stats["false_positive"] += 1
                    stats["min_incorrect_distance"] = min(stats["min_incorrect_distance"], distance)
                    stats["fp_distance_sum"] += distance
            else:
                if is_same_cow:
                    stats["false_negative"] += 1
                else:
                    stats["true_negative"] += 1
    return stats


def compute_metrics(stats, threshold):
    """
    Computes evaluation metrics from collected statistics.

    Args:
        stats (dict): Dictionary of evaluation statistics.
        threshold (float): Evaluation threshold used.

    Returns:
        dict: Dictionary of computed metrics.
    """
    tp = stats["true_positive"]
    fp = stats["false_positive"]
    tn = stats["true_negative"]
    total = stats["total"]
    accuracy = (tp + tn) / total if total > 0 else 0
    if threshold == -1:
        if tp > 0:
            average_true_positive_distance = stats["tp_distance_sum"] / tp
        else:
            average_true_positive_distance = None

        if fp > 0:
            average_false_positive_distance = stats["fp_distance_sum"] / fp
        else:
            average_false_positive_distance = None

        metrics = {
            "max_correct_distance": stats["max_correct_distance"],
            "min_incorrect_distance": stats["min_incorrect_distance"],
            "average_true_positive_distance": average_true_positive_distance,
            "average_false_positive_distance": average_false_positive_distance,
            "accuracy": accuracy,
        }
        return metrics
    else:
        fn = stats["false_negative"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate = Recall
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        balanced_acc = (tpr + tnr) / 2

        metrics = {
            "accuracy": accuracy,
            "TPR": tpr,
            "TNR": tnr,
            "balanced acc": balanced_acc,
            "precision": precision,
        }
        return metrics


def run_experiment(model_name="mobilenet", model_path="models/mobilenet.pth", mode="identification", thresholds=None):
    """
    Runs an evaluation experiment with the given model and thresholds.

    Args:
        model_name (str, optional): Name of the model architecture. Defaults to "mobilenet".
        model_path (str, optional): Path to the trained model file. Defaults to "models/mobilenet.pth".
        mode (str, optional): Evaluation mode: "identification" or "new_cows". Defaults to "identification".
        thresholds (list, optional): List of thresholds to evaluate. Defaults to None.
    """
    # Set thresholds and test folder
    if mode == "identification":
        if thresholds is None:
            thresholds = [-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        test_folder = "data/id_dataset_split/test"
    elif mode == "new_cows":
        if thresholds is None:
            thresholds = [-1, 0.7, 0.8, 0.9, 1.0, 1.1]
        test_folder = "data/id_dataset_split/newcows_test"
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'identification' or 'new_cows'.")

    # Load model
    model_class = MODEL_REGISTRY[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    for threshold in thresholds:
        stats = evaluate_model_on_dataset(model, test_folder, threshold, mode=mode)
        metrics = compute_metrics(stats, threshold)

        # Print metrics
        if threshold == -1:
            print("\n=== No threshold ===")
            distances = {
                "genuine": stats["genuine_distances"],
                "impostor": stats["impostor_distances"]
            }
            plot_genuine_impostor_histogram(distances, model_name,
                                            save_path=f"genuine_impostor_{mode}_{model_name}.pdf")
            print(f"max_correct_distance: {metrics['max_correct_distance']:.4f}")
            print(f"min_incorrect_distance: {metrics['min_incorrect_distance']:.4f}")

            avg_tp = metrics.get("average_true_positive_distance")
            avg_fp = metrics.get("average_false_positive_distance")

            print(f"average_true_positive_distance: {avg_tp:.4f}" if avg_tp is not None
                  else "average_true_positive_distance: N/A")
            print(f"average_false_positive_distance: {avg_fp:.4f}" if avg_fp is not None
                  else "average_false_positive_distance: N/A")
        else:
            print(f"\n=== Threshold: {threshold:.2f} ===")
            if mode == "identification":
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"TPR (Recall): {metrics['TPR']:.4f}")
                print(f"TNR (Specificity): {metrics['TNR']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Balanced accuracy: {metrics['balanced acc']:.4f}")
            else:
                print(f"Accuracy: {metrics['accuracy']:.4f}")
