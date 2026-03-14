# Federated Learning on COVID-19 Radiography Dataset

Comprehensive benchmarking of Federated Learning under IID and Non-IID data distributions, including client contribution analysis using Leave-One-Out (LOO) experiments, implemented using PyTorch and Flower (flwr).

## Project Overview

This project implements a complete Federated Learning (FL) experimentation pipeline using the COVID-19 Radiography Dataset.

The objective is to analyze:

* Federated Learning performance across multiple data heterogeneity scenarios
* Client contribution in federated systems
* Client benefit vs contribution relationships
* Fairness and stability metrics
* Performance comparison between isolated training and federated training

The project simulates 3 FL clients and evaluates training across 7 dataset split types representing different real-world data distributions.

## Key Features
### Federated Learning Framework

* Flower (flwr) based FL system
* FedAvg aggregation
* 3 simulated clients
* 10 training rounds per experiment

### CNN Models Evaluated

* SimpleCNN
* SimpleCNN_V2
* ResNet-18
* MobileNetV2

### Data Distribution Scenarios

Seven types of federated splits:

| Split Type | Description |
| ------------- | ------------- |
| IID_equal| Balanced IID distribution  |
| Quantity_skew  | Unequal data size across clients  |
| Label_skew              | Class imbalance between clients              |
| Feature_skew              | Feature distribution differences              |
| Dirichlet_label              | Probabilistic label imbalance              |
| Pathological              | Extreme heterogeneity              |
| Concept_shift                          | Concept distribution differences              |


### Contribution Analysis

* Leave-One-Out (LOO) experiments
* Client contribution estimation
* Client benefit evaluation
* Fairness index calculation
* Stability analysis

### Visualization & Analysis

* Accuracy/Loss plots
* Contribution vs Benefit plots
* Client comparison analysis
* Fairness and stability evaluation

## Dataset Description

The dataset contains three disease classes:

* Covid
* Normal
* Viral Pneumonia

Dataset is augmented and then split into federated client datasets.

## Dataset Directory Structure
```
Covid19-dataset/
│
├── augmented/
│   ├── Covid/
│   ├── Normal/
│   └── Viral Pneumonia/
│
├── train/
├── test/
│
├── splits/
│   ├── IID_equal/
│   ├── Quantity_skew/
│   ├── Label_skew/
│   ├── Feature_skew/
│   ├── Dirichlet_label/
│   ├── Pathological/
│   └── Concept_shift/
│
└── split_metadata.csv
```

Inside each split folder:
```
split_name/
│
├── Client-1/
├── Client-2/
└── Client-3/
    │
    ├── train/
    │   ├── Covid/
    │   ├── Normal/
    │   └── Viral Pneumonia/
    │
    └── test/
        ├── Covid/
        ├── Normal/
        └── Viral Pneumonia/
        
```

Each client has its own train and test dataset.

## Project Directory Structure
```
federated_learning_project/
│
├── notebooks/
│
│   01_c19_data_exploration.ipynb
│   02_c19_data_augmentation.ipynb
│   02.1_c19_data_demograph.ipynb
│   03_c19_data_separation.ipynb
│   04_c19_isolated_training.ipynb
│   05_c19_isolated_training_plots.ipynb
│   06_c19_federated_training.ipynb
│   07_c19_fl_contribution_setup.ipynb
│   08_c19_client_contribution_analysis.ipynb
│
│
├── src/
│
│   client.py
│   server.py
│   server_contribution.py
│   model.py
│   dataset.py
│
│   main.py
│   main_contribution.py
│
│   client_contribution_eval.py
│   contribution_summary.py
│
│   accuracy_loss_metrics_plot.py
│   client_contribution_analysis_plot.py
│   client_benefit_analysis_plot.py
│   benefit_vs_contribution_plot.py
│   benefit_gap_summary_table.py
│   fairness_index_and_stability_plot.py
│
│
├── logs/
│
│   client_*_metrics_<model>_<split>.csv
│   log_client_metrics_<model>_<split>.csv
│   log_global_metrics_<model>_<split>.csv
│
│
├── contribution_logs/
│
│   log_client_metrics_<model>_with_all_<split>.csv
│   log_client_metrics_<model>_excl_client_i_<split>.csv
│
│   log_global_metrics_<model>_with_all_<split>.csv
│   log_global_metrics_<model>_excl_client_i_<split>.csv
│
│
├── best_models/
│
│   tinycnn_model.pth
│   simplecnn_model.pth
│   simplecnn_v2_model.pth
│   resnet18_model.pth
│   mobilenetv2_model.pth
│
├── requirements.txt
└── README.md
```

## Logging System
### Training Logs (logs/)
1. Client Metrics

```client_*_metrics_<model>_<split>.csv```

Columns:
* round
* loss
* accuracy
2. Combined Client Metrics

```log_client_metrics_<model>_<split>.csv```

Columns:

* round
* client_id
* accuracy
* loss
3. Global Metrics

```log_global_metrics_<model>_<split>.csv```

Columns:

* round
* global_accuracy

### Contribution Logs

Stored in ```contribution_logs/```

These logs are generated during Leave-One-Out experiments.

**With All Clients**
```
log_client_metrics_<model>_with_all_<split>.csv
log_global_metrics_<model>_with_all_<split>.csv
```
**Excluding Client i**
```
log_client_metrics_<model>_excl_client_i_<split>.csv
log_global_metrics_<model>_excl_client_i_<split>.csv
```
## Contribution Calculation

Client contribution is estimated using Leave-One-Out (LOO).
```
Contribution_i = GlobalAccuracy(with_all_clients) − GlobalAccuracy(excluding_client_i)
```
A larger drop in global accuracy indicates a higher contribution from that client.

## Benefit Calculation

Benefit measures how much a client improves from federated training.
```
Benefit_i = FederatedAccuracy_i − IsolatedAccuracy_i
```
This shows whether FL helps each client.

## Fairness and Stability Metrics

The project evaluates:

* Jain's Fairness Index
* Stability Index
* Variance of contributions
* Contribution distribution balance

Generated using:

```fairness_index_and_stability_plot.py```
## Experiment Workflow
### Step 1 — Data Exploration
```01_c19_data_exploration.ipynb```

Dataset statistics and visualization.

### Step 2 — Data Augmentation
```02_c19_data_augmentation.ipynb```

Uses Albumentations for:

* rotations
* flips
* brightness/contrast
* noise
* elastic transforms

### Step 3 — Dataset Splitting
```03_c19_data_separation.ipynb```

Creates the 7 FL split scenarios.

### Step 4 — Isolated Training
```04_c19_isolated_training.ipynb```

Trains models locally for baseline performance.

### Step 5 — Plot Isolated Results
```05_c19_isolated_training_plots.ipynb```

### Step 6 — Federated Training
```06_c19_federated_training.ipynb```

Runs Flower-based FL training.

### Step 7 — Contribution Setup

```07_c19_fl_contribution_setup.ipynb```

Prepares LOO experiments.

### Step 8 — Contribution Analysis
```08_c19_client_contribution_analysis.ipynb```

Generates:

* contribution plots
* benefit plots
* fairness plots

## Installation

Install dependencies:

```pip install -r requirements.txt```
### Running Experiments

**Federated Training**

```python src/main.py```

**Contribution Experiments**

```python src/main_contribution.py```

**Generate Plots**

```
python src/accuracy_loss_metrics_plot.py
python src/client_contribution_analysis_plot.py
python src/client_benefit_analysis_plot.py
python src/benefit_vs_contribution_plot.py
python src/fairness_index_and_stability_plot.py
```
## Technologies Used

* Python
* PyTorch
* Flower (Federated Learning Framework)
* Albumentations
* OpenCV
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly

## Future Improvements

Potential extensions include:

* Implement FedProx / FedNova algorithms
* Increase number of simulated clients
* Add differential privacy
* Secure aggregation
* Apply to real multi-hospital datasets
* Experiment with transformer architectures

## License

MIT License.
