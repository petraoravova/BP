# author: Petra Oravová
# file: train.py
# description: Script to train a deep learning model using triplet loss for cow identification.
#              Includes training and validation loop with early stopping and model saving.
# used documentation and tutorials:
# - https://docs.pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
# - https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
# - https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from identification.triplet_dataset import TripletDataset
import torch.nn.functional as F
from identification.model import MODEL_REGISTRY


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        criterion (torch.nn.TripletMarginLoss): Loss function.
        device (torch.device): The device (CPU or GPU) on which to perform training.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    epoch_loss = 0

    for anchor, positive, negative in train_loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()

        anchor_emb = F.normalize(model(anchor), p=2, dim=1)
        positive_emb = F.normalize(model(positive), p=2, dim=1)
        negative_emb = F.normalize(model(negative), p=2, dim=1)

        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    return avg_train_loss


def validate_epoch(model, val_loader, criterion, device):
    """
    Validates the model for one epoch.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        criterion (torch.nn.TripletMarginLoss): Loss function.
        device (torch.device): The device (CPU or GPU) on which to perform validation.

    Returns:
        float: Average validation loss for the epoch.
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for anchor, positive, negative in val_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_emb = F.normalize(model(anchor), p=2, dim=1)
            positive_emb = F.normalize(model(positive), p=2, dim=1)
            negative_emb = F.normalize(model(negative), p=2, dim=1)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def train_model(model_name="mobilenet", output_name="models/best_model.pth"):
    """
    Trains the model using the specified architecture and saves the best model based on validation loss.

    Args:
        model_name (str, optional): The name of the model architecture. Defaults to "mobilenet".
        output_name (str, optional): The path where the best model weights will be saved. Defaults to
                                    "models/best_model.pth".
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 100
    BATCH_SIZE = 16
    PATIENCE = 10

    # Initialize model, loss function, optimizer and scheduler
    model_class = MODEL_REGISTRY[model_name]
    model = model_class().to(device)
    criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Prepare datasets and dataloaders
    train_dataset = TripletDataset("data/id_dataset_split/train", transform=model.transform, is_training=True)
    val_dataset = TripletDataset("data/id_dataset_split/val", transform=model.transform, is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Training
        avg_train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # Validation
        avg_val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        # Early stopping and saving the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_name)
            print("New best model saved.")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    print("Done.")


if __name__ == "__main__":
    # Entry point of the script to run training
    train_model()
