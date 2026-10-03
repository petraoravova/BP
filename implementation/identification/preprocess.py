# author: Petra Oravová
# file: preprocess.py
# description: This script splits the dataset into training, validation, and test sets with a specified ratio.
#              It also reserves a small portion of cows as "new cows" for testing.
# Used documentation and tutorials:
# - https://pypi.org/project/split-folders/

import os
import random
import shutil
import splitfolders

SPLIT_RATIO = (0.7, 0.15, 0.15)  # train, val, test
NEW_COWS_RATIO = 0.05


def split_dataset(dataset_folder="data/id_dataset_crop", output_folder="data/id_dataset_split"):
    """
    Splits the dataset into training, validation, and testing sets based on the provided ratio.
    Also, reserves a portion of cows for "new cows" testing.

    Args:
        dataset_folder (str, optional): Path to the input dataset folder containing subfolders for each cow.
                                        Defaults to "data/id_dataset_crop"
        output_folder (str, optional): Path to the output folder where the split datasets will be saved.
                                        Defaults to "data/id_dataset_split"
    """
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)

    newcows_folder = os.path.join(output_folder, "newcows_test")
    os.makedirs(newcows_folder, exist_ok=True)

    # Get all cow IDs (subfolders) in the dataset
    cow_ids = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]

    random.seed(42)
    random.shuffle(cow_ids)

    # Move the new cows to the "new cows" test folder
    num_new_cows = int(len(cow_ids) * NEW_COWS_RATIO)
    new_cows = cow_ids[:num_new_cows]
    for cow_id in new_cows:
        src = os.path.join(dataset_folder, cow_id)
        dst = os.path.join(newcows_folder, cow_id)
        shutil.move(src, dst)

    # Split the dataset into train/val/test using the specified ratio
    splitfolders.ratio(
        dataset_folder,
        output=output_folder,
        seed=42,
        ratio=SPLIT_RATIO,
        group_prefix=None
    )

    # Copy the new cows back into the original dataset folder after splitting
    for cow_id in new_cows:
        src = os.path.join(newcows_folder, cow_id)
        dst = os.path.join(dataset_folder, cow_id)
        shutil.copytree(src, dst)

    print(f"Dataset split completed: {len(cow_ids) - len(new_cows)} cows split into train/val/test, {len(new_cows)} "
          f"cows reserved for new cow testing.")


if __name__ == "__main__":
    # Entry point of the script
    # Calling the function to split the dataset
    split_dataset(dataset_folder="../data/id_dataset_crop", output_folder="../data/id_dataset_split")
