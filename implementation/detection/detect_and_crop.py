# author: Petra Oravová
# file: detect_and_crop.py
# description: Script to detect objects in images using a YOLO11n model and crop the best detected bounding box.
# Used documentation and tutorials:
# - https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
# - https://learnopencv.com/image-resizing-with-opencv/
# - https://learnopencv.com/cropping-an-image-using-opencv/

import os
import cv2
from ultralytics import YOLO
import shutil


def detect_and_crop(input_path, resize_dim=(256, 256)):
    """
    Loads an image, detects objects using a YOLO model, crops the most confident detection,
    and resizes it to the specified dimensions.

    Args:
        input_path (str): Path to the input image.
        resize_dim (Tuple[int, int]): Tuple (width, height) to resize the cropped image to.

    Returns:
        Tuple[Optional[array], Optional[Tuple[int, int, int, int]]]: Cropped and resized image, and bounding box
                                        coordinates (x1, y1, x2, y2). If no object is detected, returns (None, None).
    """
    model = YOLO("models/yolo11n_model.pt")
    img = cv2.imread(input_path)
    results = model(input_path, imgsz=640)[0]

    if len(results.boxes) > 0:
        # Get the most confident bounding box
        best_box = max(results.boxes, key=lambda b: b.conf[0])
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        cropped = img[y1:y2, x1:x2]
        cropped = cv2.resize(cropped, resize_dim)

        return cropped, (x1, y1, x2, y2)

    return None, None


def process_folder(input_folder="data/id_dataset_raw", output_folder="data/id_dataset_crop"):
    """
    Processes all images in the input folder, applies detection and cropping,
    and saves the cropped images in the output folder preserving the folder structure.

    Args:
        input_folder (str, optional): Path to the folder containing raw images. Defaults to "data/id_dataset_raw".
        output_folder (str, optional): Path to the folder where cropped images will be saved.
                                        Defaults to "data/id_dataset_crop"
    """
    all_img = 0
    crop_img = 0

    # Remove the output folder if it already exists
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                all_img += 1
                input_path = os.path.join(root, file)

                # Preserve subfolder structure in the output folder
                relative_path = os.path.relpath(input_path, input_folder)
                output_subfolder = os.path.join(output_folder, os.path.dirname(relative_path))
                os.makedirs(output_subfolder, exist_ok=True)

                # Detect and crop
                cropped = detect_and_crop(input_path)[0]
                if cropped is not None:
                    crop_img += 1
                    output_path = os.path.join(output_subfolder, f"{os.path.splitext(file)[0]}_crop.png")
                    cv2.imwrite(output_path, cropped)

    print(f"All images: {all_img}. Not detected: {all_img - crop_img}")


if __name__ == "__main__":
    # Entry point of the script
    # Defines input and output directories and starts the processing of images
    # Each image is processed using YOLO detection; the most confident bounding box is cropped and saved

    input_folder = "../data/id_dataset_raw"
    output_folder = "../data/id_dataset_crop"

    process_folder(input_folder, output_folder)
