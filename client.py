
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.client import NumPyClient
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import csv

sys.path.append('/content/drive/MyDrive/Colab Notebooks')
from dataset import get_dataloaders
from model import TinyCNN, SimpleCNN_V2, ResNet18, MobileNetV2

LOG_DIR = "/content/drive/MyDrive/Colab Notebooks/logs"
USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"
DEVICE = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

def get_model(model_name, num_classes=3):
    if model_name == "tinycnn":
        return TinyCNN(num_classes).to(DEVICE)
    elif model_name == "simplecnn":
        return SimpleCNN_V2(num_classes).to(DEVICE)
    elif model_name == "resnet18":
        return ResNet18(num_classes).to(DEVICE)
    elif model_name == "mobilenetv2":
        return MobileNetV2(num_classes).to(DEVICE)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

class FlowerClient(NumPyClient):
    def __init__(self, model_name, data_path, client_id, split_name="default"):
        self.model = get_model(model_name)
        self.train_loader, self.val_loader = get_dataloaders(data_path)
        self.client_id = client_id
        self.split_name = split_name

        # Initialize log file
        os.makedirs(LOG_DIR, exist_ok=True)

        # Initialize log file
        self.log_file = f"{LOG_DIR}/client_{client_id}_metrics_{model_name}_{split_name}.csv"
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "loss", "accuracy"])

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters, config=None):
        # Convert if needed
        if hasattr(parameters, "tensors"):  # it's a Parameters object
            parameters = parameters_to_ndarrays(parameters)

        # Load into state_dict
        state_dict = self.model.state_dict()
        for k, val in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(val)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(1):
            for x, y in self.train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                y_hat = self.model(x)
                loss = F.cross_entropy(y_hat, y)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        self.model.eval()

        correct = 0
        loss = 0.0
        total = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_hat = self.model(x)
                loss += F.cross_entropy(y_hat, y, reduction='sum').item()
                correct += (y_hat.argmax(1) == y).sum().item()
                total += y.size(0)

        avg_loss = loss / total
        accuracy = correct / total
        round_number = config.get("server_round", 0)

        # Log results to CSV
        if config.get("server_side", False):
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([round_number, avg_loss, accuracy])

        return avg_loss, total, {"accuracy": accuracy}
