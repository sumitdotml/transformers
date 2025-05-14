"""
Dataset loading and processing utilities.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    """
    Template dataset class.
    """

    def __init__(self, data_path, transform=None):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the data
            transform: Optional transform to apply to the data
        """
        self.data_path = data_path
        self.transform = transform
        # TODO: Load your data here
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # TODO: Process your data item here

        if self.transform:
            item = self.transform(item)

        return item


def get_dataloader(data_path, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a DataLoader for the dataset.

    Args:
        data_path: Path to the data
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker threads for loading data

    Returns:
        DataLoader instance
    """
    dataset = CustomDataset(data_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader
