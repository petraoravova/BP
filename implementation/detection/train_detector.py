# author: Petra Oravová
# file: train_detector.py
# description: Script for training and testing a YOLO11n object detector using the Ultralytics library.
# The script assumes a properly configured dataset YAML file at data/yolo_dataset/data.yaml.
# Documentation used: https://docs.ultralytics.com/models/yolo11/

from ultralytics import YOLO


def train_detector():
    """
    Trains a YOLO11n detector using the pretrained model.
    Saves the trained model to the default output directory (runs/detect/train).
    """
    model = YOLO("yolo11n.pt")
    model.train(data="data/yolo_dataset/data.yaml", epochs=1000)


def test_detector(model_name="models/yolo11n_model.pt"):
    """
    Tests a trained YOLO11n model on the test dataset.

    Args:
        model_name (str, optional): Path to the trained model file. Defaults to "models/yolo11n_model.pt".
    """
    model = YOLO(model_name)
    model.val(data="data/yolo_dataset/data.yaml", split="test")


if __name__ == "__main__":
    # Entry point of the script to run training and testing
    # Modify as needed depending on whether you want to run both or only one step

    train_detector()
    test_detector(model_name="../models/yolo11n_model.pt")
