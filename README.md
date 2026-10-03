# Identification of Animals Using Artificial Intelligence
**Bachelor Thesis 2025**  
**Author:** Petra Oravová (`xoravo01`)  
**Supervisor:** Ing. Tomáš Goldmann, Ph.D.  
**Faculty of Information Technology, Brno University of Technology**

This repository contains all components of a bachelor's thesis focused on animal (specifically cow) identification using artificial intelligence. It includes the implementation (data processing, model training, inference, and GUI), full technical documentation, and the thesis itself in both LaTeX and PDF formats.

## Directory Structure

The main folder `xoravo01-Identifikace-zvere-pomoci-AI` is organized as follows:

```
xoravo01-Identifikace-zvere-pomoci-AI/
├── README.md                 # This file
├── final_thesis.pdf          # Bachelor thesis in PDF format (officially named xoravo01-Identifikace-zvere-pomoci-AI.pdf)
├── thesis/                   # LaTeX source files of the thesis
├── implementation/           # Source code and project implementation
│   ├── main.py               # Main script with CLI interface
│   ├── requirements.txt      # Python dependencies
│   ├── documentation.pdf     # Technical documentation of the system
│   ├── detection/            # Code related to object detection
│   ├── identification/       # Code related to individual animal identification
│   ├── data/                 # Training and testing data
│   ├── models/               # Pre-trained neural network weights
│   ├── db/                   # Database code and backups
│   └── app/                  # GUI application code
```

### Directory Details

- #### models/
   Contains pre-trained neural network weights:
    
       mobilenet.pth – MobileNet_V_Large model for animal identification
       resnet.pth – ResNet-50 model for animal identification
       efficientnet.pth – EfficientNet_B1 model for animal identification
       yolo11n_model.pt – YOLO11n model for muzzle detection


- #### db/backups/
    PostgreSQL database dumps for each embedding model:
    
        mobilenet.dump
        resnet.dump
        efficientnet.dump
    These can be used to fill the database without image processing.
    

- #### data/
    Contains datasets for detection and identification:
    
        yolo_dataset/ – Images and labels for training the YOLO11 detector
        id_dataset_raw/ – Raw images used to build the identification dataset
        id_dataset_split/ – Final train/val/test/newcows_test split for identification models
        pairs.json – Evaluation file for fair testing of all embedding models


## Prerequisites
- The project was developed and tested on:
  - OS: Ubuntu 24.04.2 LTS (recommended)
  - Python: 3.12.3

All commands below should be executed from within the `implementation` directory.
- It is recommended to use a virtual environment:
  
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

- Install required libraries:
  #### Installation Notes: OpenCV and Ultralytics
  
  To avoid Qt plugin conflicts (especially with `PyQt5`), this project uses `opencv-python-headless` instead of the default `opencv-python`.
  
  Because `ultralytics` depends on `opencv-python`, which causes compatibility issues with `PyQt5`, please follow this installation order:

    ```bash
    # 1. Install Ultralytics without dependencies to avoid installing opencv-python
    pip install --no-deps ultralytics==8.3.133
      
    # 2. Install all remaining dependencies (including opencv-python-headless)
    pip install -r requirements.txt
  ```

### Database Setup

This project uses a **PostgreSQL** database for storing and managing data related to cow identification. The connection 
to the database is made directly in the code. It also requires the pgvector PostgreSQL extension for handling 
vector data (e.g., similarity search between embeddings).

#### Connection Details
The database connection is established using the following parameters:
- **Database Name**: `cow_identification`
- **User**: `postgres`
- **Password**: `951753456`
- **Host**: `localhost`
- **Port**: `5432`

These values are hardcoded into the code. If you wish to change them, you will need to modify the following files:

1. **`db/connection.py`**:
   - The connection is made in the `get_connection` and `create_database` functions. You can change the database credentials in these functions.
   
2. **`db/backup.py`**:
   - If you want to perform database backups, the password is set in this file using the `os.environ['PGPASSWORD']` method.

#### Setting Up the Database

- **Install PostgreSQL** (if not already installed).

        sudo apt install postgresql

- **Install pgvector PostgreSQL extension** (replace 16 with your PostgreSQL version):

       sudo apt install postgresql-16-pgvector 


- **Create and fill the database**:
  
    If the database `cow_identification` does not exist, you can create it manually or run the `fill` or `import` mode 
        in the program, which will automatically create the database if it does not exist and fill it with data. Simply run:
     ```bash
     python main.py --mode fill --model <your_model_name> --model_path <path_to_model>
     ```
     Or use:
     ```bash
     python main.py --mode import --input <your_filename>
     ```

## Running the Program

Use the `main.py` script to run different parts of the application via command-line interface:

```bash
python main.py --mode <mode> [additional arguments]
```

### Available Modes

| Mode            | Description                                           |
|-----------------|-------------------------------------------------------|
| `test_prepare`  | Generate and save image pairs for verification tasks. |
| `preprocess`    | Preprocess and split the dataset.                     |
| `id_train`      | Train an animal identification model.                 |
| `detect_train`  | Train an object detector.                             |
| `detect_test`   | Test the object detection model.                      |
| `crop`          | Run object detection and crop detected animals.       |
| `fill`          | Use a model to fill a database.                       |
| `export`        | Export the database to a file.                        |
| `import`        | Import a database from a file.                        |
| `test`          | Run a test evaluation of the identification model.    |
| `infer`         | Run inference on a single image.                      |
| `app`           | Launch the graphical user interface.                  |

### Common Arguments

- `--model` – Model architecture name (`resnet`, `efficientnet`, `mobilenet`).
- `--output` – Output file name/path (used for saving models or export).
- `--input` – Input file name/path (used for import).
- `--model_path` – Path to pre-trained model weights (e.g. `models/mobilenet.pth`).
- `--image` – Path to input image (used in inference mode).

### Examples

Train an identification model:
```bash
python main.py --mode id_train --model resnet --output my_model.pth
```

Run inference on an image:
```bash
python main.py --mode infer --image data/id_dataset_raw/1/A_0.jpg --model resnet --model_path models/resnet.pth
```

Launch the GUI application:
```bash
python main.py --mode app
```

### Detection
To train or test the object detection model (YOLO11n), use the following modes:

```bash
# Train YOLO detector
python main.py --mode detect_train

# Test YOLO detector
python main.py --mode detect_test [--model_path <path_to_model.pt>]
```
- model_path (optional): Path to a custom YOLO model file. If not provided, a default model will be used.
- Before training or testing the YOLO model, make sure to update the `data.yaml` file located in implementation/data/yolo_dataset/ with the absolute path to your dataset. YOLO requires absolute paths to function correctly.


## Datasets

The datasets used for this project are based on open-source animal datasets, including:

- **[DVC-5](https://universe.roboflow.com/dvc/dvc-5)** 
- **[Muzzle detection1](https://universe.roboflow.com/muzzle-detection-ev5cg/muzzle-detection1)** 
- **[Cow](https://universe.roboflow.com/robo-lpmzn/cow-rpgxt)** 
- **[Cattle Muzzle Images Dataset (CMID)](https://www.kaggle.com/datasets/kollabathulakaushik/cattle-images-db-for-muzzle-based-identification)** 
- **[Muzzle-268](https://doi.org/10.5281/zenodo.6324361)**
- **[Cows Frontal Face Dataset (CFFD)](https://doi.org/10.5281/zenodo.10535934)**
-  **[Muzzle Cow New Dataset (MCND)](https://doi.org/10.5281/zenodo.7988559)**
-  **[Muzzle-211385 (Kaggle)](https://www.kaggle.com/datasets/norashimah/muzzle-211385)**
-  **[Roboflow](https://universe.roboflow.com/muzzle-identification9161/cattle-identification)**

Details about the use and adaptation of the datasets are explained in the `xoravo01-Identifikace-zvere-pomoci-AI.pdf`.

## Documentation

- `xoravo01-Identifikace-zvere-pomoci-AI.pdf`: Complete bachelor thesis describing the system.
- `implementation/documentation.pdf`: Technical documentation contains information about the structure, files, and functions within the system.
- LaTeX sources for the thesis can be found in the `thesis/` directory.

