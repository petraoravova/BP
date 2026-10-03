# author: Petra Oravová
# file: utils.py
# description: This script provides functions for managing cow embeddings in a PostgreSQL database.
#              It includes functionality to insert a cow's embedding, identify a cow by its embedding,
#              check if a cow exists, and fill the database with embeddings from a trained model.
# used documentation and tutorials:
# - https://www.postgresql.org/docs/current/tutorial-sql.html
# - https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html


from db.connection import execute_query
from db.connection import create_table, create_database
from identification.model import MODEL_REGISTRY
import torch
import os


def insert_cow(cow_id, embedding):
    """
    Inserts a cow's ID and its corresponding embedding into the 'cow_embeddings' table.

    Args:
        cow_id (str): The unique identifier for the cow.
        embedding (list): The embedding vector for the cow.
    """
    query = """
    INSERT INTO cow_embeddings (cow_id, embedding)
    VALUES (%s, %s)
    """
    execute_query(query, (cow_id, embedding))
    print(f"Cow {cow_id} done.")


def identify_cow(embedding, threshold=0.3):
    """
    Identifies the closest matching cow in the database based on the provided embedding.

    Args:
        embedding (list): The embedding to compare against existing cow embeddings.
        threshold (float, optional): The similarity threshold below which no match will be returned.
                           If threshold is -1, no similarity limit is applied and the closest match
                           will always be returned. Defaults to 0.3.

    Returns:
        Tuple[Optional[str], Optional[float]]: The cow ID and distance if a match is found within the threshold,
                                                else None.
    """
    query = """
    SELECT cow_id, embedding <-> %s::vector AS distance
    FROM cow_embeddings
    ORDER BY distance ASC
    LIMIT 1;
    """
    result = execute_query(query, (embedding,), fetchone=True)
    if result and (threshold < 0 or result[1] <= threshold):
        return result[0], result[1]
    else:
        return None, result[1]


def cow_exists(cow_id):
    """
    Checks if a cow with the given ID already exists in the 'cow_embeddings' table.

    Args:
        cow_id (str): The unique identifier for the cow to check.

    Returns:
        bool: True if the cow exists, otherwise False.
    """
    query = "SELECT EXISTS(SELECT 1 FROM cow_embeddings WHERE cow_id = %s)"
    result = execute_query(query, (cow_id,), fetchone=True)
    return result[0] if result else False


def fill(model_name="mobilenet", model_path="models/mobilenet.pth"):
    """
    Fills the database with cow embeddings using a trained model. First, the database and table are created
    if they don't exist. Then, for each image in the training dataset, an embedding is generated and inserted.

    Args:
        model_name (str, optional): The name of the model to use for embedding extraction. Defaults to "mobilenet".
        model_path (str, optional): The file path to the pre-trained model weights. Defaults to "models/mobilenet.pth".
    """
    # Prepare database
    create_database()
    create_table()
    delete_query = "DELETE FROM cow_embeddings"
    execute_query(delete_query)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = MODEL_REGISTRY[model_name]
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path))

    # Loop through dataset and insert embeddings
    for folder in os.listdir("data/id_dataset_split/train"):
        folder_path = os.path.join("data/id_dataset_split/train", folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and file.lower().endswith('.png'):
                    embedding = model.get_embedding(file_path, device)
                    insert_cow(folder, embedding)

    print("All cows in database.")
