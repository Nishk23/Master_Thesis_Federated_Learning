
import os
import shutil
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataloaders(data_dir, batch_size=32, val_split=0.2, seed=42):

    for class_dir in os.listdir(data_dir):
        full_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(full_path) and len(os.listdir(full_path)) == 0:
            print(f" Removing empty folder: {full_path}")
            os.rmdir(full_path)
   
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size

    torch.manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader
