
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks')

import os
import csv
import flwr as fl
from flwr.common import Context
from flwr.simulation import start_simulation
from client import FlowerClient

# Constants
LOG_DIR = "/content/drive/MyDrive/Colab Notebooks/logs"
DATA_BASE = "/content/drive/MyDrive/Colab Notebooks/Covid19-dataset/splits"
CLIENTS = []

os.makedirs(LOG_DIR, exist_ok=True)

# ------------------------------
# Evaluation Function
# ------------------------------
def get_eval_fn(model_name, split_name):
    def evaluate(server_round, parameters, config):
        total_correct = 0
        total_samples = 0
        round_logs = []

        for i, client in enumerate(CLIENTS):
            # print(f"Client {i} evaluate arg count:", client.evaluate.__code__.co_argcount)
            loss, num_samples, metrics = client.evaluate(parameters, config)
            acc = metrics.get("accuracy", 0.0)
            total_correct += acc * num_samples
            total_samples += num_samples
            round_logs.append([server_round, i, acc, loss])

        # Log client metrics
        log_client_path = f"{LOG_DIR}/log_client_metrics_{model_name}_{split_name}.csv"
        with open(log_client_path, "a", newline="") as f:
            writer = csv.writer(f)
            if os.stat(log_client_path).st_size == 0:
                writer.writerow(["round", "client_id", "accuracy", "loss"])
            writer.writerows(round_logs)

        # Log global metrics
        global_acc = total_correct / total_samples if total_samples > 0 else 0.0
        log_global_path = f"{LOG_DIR}/log_global_metrics_{model_name}_{split_name}.csv"
        with open(log_global_path, "a", newline="") as f:
            writer = csv.writer(f)
            if os.stat(log_global_path).st_size == 0:
                writer.writerow(["round", "global_accuracy"])
            writer.writerow([server_round, global_acc])

        return 0.0, {"accuracy": global_acc}

    return evaluate

# ------------------------------
# Custom Strategy Subclass
# ------------------------------
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, eval_fn, **kwargs):
        super().__init__(**kwargs)
        self._eval_fn_custom = eval_fn

    def evaluate(self, server_round, parameters, config=None):  # Make config optional
        if config is None:
            config = {}  # Provide empty config if missing
        config["server_round"] = server_round
        config["server_side"] = True
        return self._eval_fn_custom(server_round, parameters, config)

# ------------------------------
# Client Setup
# ------------------------------
def get_clients(model_name, split_name):
    global CLIENTS
    clients = []
    for i in range(3):
        data_path = os.path.join(DATA_BASE, split_name, f"Client-{i+1}")
        # print(f"✅ Initialized client {i} with path: {data_path}")
        client = FlowerClient(model_name, data_path, i, split_name)
        clients.append(client)
    CLIENTS = clients
    return clients

# ------------------------------
# Run Simulation
# ------------------------------
def run_simulation(model_name, split_name):
    clients = get_clients(model_name, split_name)
    eval_fn = get_eval_fn(model_name, split_name)

    strategy = CustomFedAvg(
        eval_fn=eval_fn,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3
    )

    def client_fn(cid: str):
      return clients[int(cid)]

    start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )

# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <model_name> <split_name>")
        sys.exit(1)

    model = sys.argv[1].lower()
    split = sys.argv[2]
    run_simulation(model, split)
