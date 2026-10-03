# author: Petra Oravová
# file: main.py
# description: Main control script for the project
#
# This script serves as the entry point for various system functions:
# - Dataset preprocessing
# - Model training (identification, detection)
# - Image cropping using object detection
# - Model testing and inference
# - Database operations (import/export/fill)
# - Launching the graphical user interface
# Usage is controlled via CLI arguments using the `--mode` flag.
#
# used documentation and tutorials:
# - https://docs.python.org/3/library/argparse.html

import argparse
import sys
from PyQt5.QtWidgets import QApplication
from identification.preprocess import split_dataset
from identification.train import train_model
from identification.infer import infer
from identification.test import run_experiment, generate_and_save_pairs
from db.utils import fill
from detection.train_detector import train_detector, test_detector
from detection.detect_and_crop import process_folder
from db.backup import export_database, import_database
from app.gui import CowApp, APP_STYLE


def run_test_prepare(args):
    """Generates and saves image pairs for testing."""
    print("Running test_prepare...")
    if args.output:
        generate_and_save_pairs(save_path=args.output)
    else:
        generate_and_save_pairs()


def run_preprocess(args):
    """Runs dataset splitting."""
    print("Running dataset preprocessing...")
    split_dataset()


def run_id_train(args):
    """Trains an embedding model."""
    print("Training identification model...")
    if args.model:
        if args.output:
            train_model(model_name=args.model, output_name=args.output)
        else:
            train_model(model_name=args.model)
    else:
        if args.output:
            train_model(output_name=args.output)
        else:
            train_model()


def run_detect_train(args):
    """Trains an object detection model."""
    print("Training object detector...")
    train_detector()


def run_detect_test(args):
    """Tests a trained object detector."""
    print("Testing object detector...")
    if args.model_path:
        test_detector(model_name=args.model_path)
    else:
        test_detector()


def run_crop(args):
    """Runs object detection and crops detected regions from images."""
    print("Running object detection and image cropping...")
    process_folder()


def run_fill(args):
    """Fills the database using an embedding model."""
    if not args.model or not args.model_path:
        raise ValueError("For 'fill', --model and --model_path are required.")
    print("Filling database...")
    fill(model_name=args.model, model_path=args.model_path)


def run_export(args):
    """Exports the database to a file."""
    if not args.output:
        raise ValueError("For 'export', --output is required.")
    print("Exporting database...")
    export_database(output_filename=args.output)


def run_import(args):
    """Imports a database from a file."""
    if not args.input:
        raise ValueError("For 'import', --input is required.")
    print("Importing database...")
    import_database(input_filename=args.input)


def run_test(args):
    """Runs an experiment using a trained model."""
    mode = "identification"
    if args.settings:
        mode = args.settings
    if args.model:
        if args.model_path:
            run_experiment(model_name=args.model, model_path=args.model_path, mode=mode)
        else:
            run_experiment(model_name=args.model, mode=mode)
    else:
        if args.model_path:
            run_experiment(model_path=args.model_path, mode=mode)
        else:
            run_experiment(mode=mode)


def run_infer(args):
    """Runs inference on a single image."""
    if not args.image:
        raise ValueError("For 'infer', --image is required.")
    print("Running inference...")
    if args.model:
        if args.model_path:
            infer(args.image, model_name=args.model, model_path=args.model_path)
        else:
            infer(args.image, model_name=args.model)
    else:
        if args.model_path:
            infer(args.image, model_path=args.model_path)
        else:
            infer(args.image)


def run_app(args):
    """Launches the graphical user interface."""
    print("Launching GUI application...")
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    window = CowApp()
    window.show()
    sys.exit(app.exec_())


# Mapping of mode names to functions with descriptions
MODES = {
    "test_prepare": (run_test_prepare, "Generate and save image pairs for testing."),
    "preprocess": (run_preprocess, "Preprocess and split the dataset."),
    "id_train": (run_id_train, "Train an identification model."),
    "detect_train": (run_detect_train, "Train an object detector."),
    "detect_test": (run_detect_test, "Test the trained object detector."),
    "crop": (run_crop, "Detect and crop images using a trained detector."),
    "fill": (run_fill, "Fill database using a model."),
    "export": (run_export, "Export database to a file."),
    "import": (run_import, "Import database from a file."),
    "test": (run_test, "Run a test experiment with a trained model."),
    "infer": (run_infer, "Run inference on a single image."),
    "app": (run_app, "Launch the graphical user interface."),
}


def main():
    """Main entry point for parsing arguments and executing the selected mode."""
    parser = argparse.ArgumentParser(
        description="Multi-purpose tool for dataset processing, training, inference, and app launching.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--mode", type=str, choices=MODES.keys(), required=True,
        help="Select mode of operation. Options:\n" + "\n".join(
            f"  {k:13} - {desc}" for k, (_, desc) in MODES.items()
        )
    )
    parser.add_argument("--image", type=str, help="Path to image file (used in 'infer' mode).")
    parser.add_argument("--model", type=str, help="Model architecture name ('resnet', 'mobilenet', 'efficientnet').")
    parser.add_argument("--output", type=str, help="Output file or model name/path.")
    parser.add_argument("--input", type=str, help="Input file name.")
    parser.add_argument("--model_path", type=str, help="Path to trained model weights.")
    parser.add_argument("--settings", type=str, help="Mode for testing ('identification', 'new_cows')")

    args = parser.parse_args()

    print(f"Selected mode: {args.mode}")
    func, _ = MODES[args.mode]
    func(args)


if __name__ == "__main__":
    main()
