
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks')

import os
from flwr.simulation import start_simulation
from client import FlowerClient
from flwr.server import ServerConfig
from server_contribution import get_eval_fn, CustomFedAvg

# Paths
DATA_BASE = "/content/drive/MyDrive/Colab Notebooks/Covid19-dataset/splits"
LOG_DIR = "/content/drive/MyDrive/Colab Notebooks/logs/contribution_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Get clients (with optional exclusion)
def get_clients(model_name, split_name, exclude_client_id=None):
    clients = []
    for i in range(3):
        if exclude_client_id is not None and i == exclude_client_id:
            continue
        data_path = os.path.join(DATA_BASE, split_name, f"Client-{i+1}")
        clients.append(FlowerClient(model_name, data_path, i, split_name))
    return clients

# Run FL for baseline or excluded client
def run_simulation(model_name, split_name, exclude_client_id=None, custom_tag=""):
    clients = get_clients(model_name, split_name, exclude_client_id)
    tag = custom_tag if custom_tag else split_name
    eval_fn = get_eval_fn(model_name, tag, clients, split_name)

    strategy = CustomFedAvg(
        eval_fn=eval_fn,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(clients),
        min_evaluate_clients=len(clients),
        min_available_clients=len(clients)
    )

    def client_fn(cid: str):
        return clients[int(cid)]

    start_simulation(
        client_fn=client_fn,
        num_clients=len(clients),
        config=ServerConfig(num_rounds=10),
        strategy=strategy
    )

# CLI entry
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main_contribution.py <model_name> <split_name> contribution")
        sys.exit(1)

    model = sys.argv[1].lower()
    split = sys.argv[2]
    mode = sys.argv[3].lower()

    if mode == "contribution":
        print("🔁 Running baseline FL with all clients...")
        run_simulation(model, split, exclude_client_id=None, custom_tag="with_all")

        for cid in range(3):
            print(f"🚫 Running FL excluding client {cid}...")
            run_simulation(model, split, exclude_client_id=cid, custom_tag=f"excl_client_{cid}")
