"""
Training utilities for neural network models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os


class Trainer:
    """
    Trainer class for neural network models.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        criterion=None,
        optimizer=None,
        lr=1e-3,
        device=None,
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: Adam)
            lr: Learning rate (default: 1e-3)
            device: Device to use for training (default: cuda if available, else cpu)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Set device
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model = self.model.to(self.device)

        # Set criterion
        self.criterion = criterion
        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss()

        # Set optimizer
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Initialize tracking variables
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0

        with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}") as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(data)

                # Compute loss
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()

                # Update weights
                self.optimizer.step()

                # Update total loss
                total_loss += loss.item()

                # Update progress bar
                pbar.set_postfix({"loss": loss.item()})

        # Compute average loss
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self):
        """
        Validate the model.

        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)

                # Compute loss
                loss = self.criterion(output, target)

                # Update total loss
                total_loss += loss.item()

        # Compute average loss
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)

        return avg_loss

    def train(self, num_epochs, save_dir=None):
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train for
            save_dir: Directory to save model checkpoints (optional)

        Returns:
            Training history (train_losses, val_losses)
        """
        # Create save directory if it doesn't exist
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_loss = self.train_epoch()
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")

            # Save checkpoint
            if save_dir is not None:
                self.save_checkpoint(os.path.join(save_dir, f"epoch_{epoch+1}.pt"))

        # Save final model
        if save_dir is not None:
            self.save_checkpoint(os.path.join(save_dir, "final_model.pt"))

        return {"train_losses": self.train_losses, "val_losses": self.val_losses}

    def save_checkpoint(self, path):
        """
        Save model checkpoint.

        Args:
            path: Path to save the checkpoint
        """
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            },
            path,
        )

    def load_checkpoint(self, path):
        """
        Load model checkpoint.

        Args:
            path: Path to the checkpoint
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
