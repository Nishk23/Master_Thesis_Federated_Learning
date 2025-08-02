
import os
import csv
import flwr as fl

LOG_DIR = "/content/drive/MyDrive/Colab Notebooks/logs/contribution_logs"

def get_eval_fn(model_name, tag, clients, split_name):
    def evaluate(server_round, parameters, config):
        total_correct, total_samples = 0, 0
        round_logs = []

        for i, client in enumerate(clients):
            loss, num_samples, metrics = client.evaluate(parameters, config)
            acc = metrics.get("accuracy", 0.0)
            total_correct += acc * num_samples
            total_samples += num_samples
            round_logs.append([server_round, i, acc, loss])

        # Log client-level accuracy
        log_client_path = f"{LOG_DIR}/log_client_metrics_{model_name}_{tag}_{split_name}.csv"
        with open(log_client_path, "a", newline="") as f:
            writer = csv.writer(f)
            if os.stat(log_client_path).st_size == 0:
                writer.writerow(["round", "client_id", "accuracy", "loss"])
            writer.writerows(round_logs)

        # Log global accuracy
        global_acc = total_correct / total_samples if total_samples > 0 else 0.0
        log_global_path = f"{LOG_DIR}/log_global_metrics_{model_name}_{tag}_{split_name}.csv"
        with open(log_global_path, "a", newline="") as f:
            writer = csv.writer(f)
            if os.stat(log_global_path).st_size == 0:
                writer.writerow(["round", "global_accuracy"])
            writer.writerow([server_round, global_acc])

        return 0.0, {"accuracy": global_acc}

    return evaluate

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, eval_fn, **kwargs):
        super().__init__(**kwargs)
        self._eval_fn_custom = eval_fn

    def evaluate(self, server_round, parameters, config=None):
        if config is None:
            config = {}
        config["server_round"] = server_round
        config["server_side"] = True
        return self._eval_fn_custom(server_round, parameters, config)
