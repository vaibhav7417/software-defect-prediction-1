# Federated Learning for Software Defect Prediction using FedProx

## Overview

This project implements a **Federated Learning (FL) framework** for **Software Defect Prediction (SDP)** using a deep neural network and the **FedProx optimization algorithm**. The system simulates a federated learning environment where multiple datasets act as independent clients that collaboratively train a shared global model.

Instead of centralizing data, each client trains a model locally and sends only model updates to a central server. The server aggregates these updates using **Federated Averaging (FedAvg)** to improve the global model.

The goal is to improve defect prediction performance while preserving **data locality and privacy**.

---

## Datasets

The project uses three publicly available **NASA software defect datasets**:

- CM1
- KC1
- PC1

Each dataset acts as a **separate federated client**.

Each dataset contains:

- Software metrics as input features
- A binary label indicating **defective (1)** or **non-defective (0)** software modules

---

## Data Preprocessing

Several preprocessing techniques are applied before training.

### 1. ADASYN Oversampling
The datasets are highly imbalanced.  
ADASYN is used to generate synthetic samples for the minority class to improve learning.

### 2. Polynomial Feature Expansion
Interaction features are generated using polynomial expansion to capture relationships between software metrics.

### 3. Feature Normalization
All features are normalized using **StandardScaler** to improve neural network training stability.

---

## Model Architecture

The model is implemented using **PyTorch** and consists of a deep fully connected neural network.

Architecture:

Input Layer  
→ Linear (Input → 512)  
→ Batch Normalization  
→ ReLU  
→ Dropout (0.03)

→ Linear (512 → 256)  
→ Batch Normalization  
→ ReLU

→ Linear (256 → 128)  
→ Batch Normalization  
→ ReLU

→ Linear (128 → 64)  
→ ReLU

→ Linear (64 → 32)  
→ ReLU

→ Linear (32 → 2)

The final layer performs **binary classification**.

---

## Federated Learning Process

The training process simulates a federated learning system.

1. Initialize a **global model**.
2. Send the global model to all clients.
3. Each client performs **local training** on its dataset.
4. Local models return updated weights.
5. Server aggregates weights using **Federated Averaging (FedAvg)**.
6. The updated global model is redistributed.

This process is repeated for **18 communication rounds**.

---

## FedProx Regularization

To handle heterogeneity across clients, the project uses **FedProx**.

FedProx introduces a proximal regularization term that restricts local model updates from deviating too far from the global model.

Loss Function:

Local Loss + FedProx Regularization

This improves convergence and stability when datasets differ significantly.

---

## Handling Class Imbalance

Three techniques are used to address class imbalance:

- ADASYN oversampling
- Class-weighted loss
- Weighted random sampling

These methods ensure better learning of defective modules.

---

## Optimization

The following optimization techniques are used:

Optimizer  
AdamW

Learning Rate Scheduler  
Cosine Annealing

Loss Function  
CrossEntropyLoss with label smoothing

---

## Evaluation Metrics

Model performance is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score

Evaluation is performed separately for each dataset.

---

## Project Workflow

```
Load Dataset
     │
     ▼
ADASYN Oversampling
     │
     ▼
Polynomial Feature Expansion
     │
     ▼
Feature Normalization
     │
     ▼
Initialize Global Model
     │
     ▼
Federated Training (18 Rounds)
     │
     ├── Local Training (FedProx)
     └── Federated Averaging
     │
     ▼
Global Model Update
     │
     ▼
Model Evaluation
```

---

## Installation

Install required dependencies:

```bash
pip install torch pandas numpy scikit-learn imbalanced-learn
```

---

## Running the Project

1. Place the datasets in your working directory:

```
cm1.csv
kc1.csv
pc1.csv
```

2. Update dataset paths in the script if required.

3. Run the training script:

```bash
python federated_training.py
```

---

## Example Output

```
--- Round 1 ---
--- Round 2 ---
...
--- Round 18 ---

CM1 - Accuracy: 0.8642 Precision: 0.8421 Recall: 0.8210 F1-score: 0.8314
KC1 - Accuracy: 0.9091 Precision: 0.8972 Recall: 0.8845 F1-score: 0.8908
PC1 - Accuracy: 0.9023 Precision: 0.8890 Recall: 0.8761 F1-score: 0.8825

Federated Learning Training & Evaluation Complete!
```

---

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn

---

## Motivation

Traditional software defect prediction models require centralized data collection. However, data sharing may not always be possible due to privacy and security constraints.

Federated Learning enables collaborative model training without sharing raw data, making it suitable for **cross-project software defect prediction**.

---

## Future Work

Possible improvements include:

- Integrating contrastive learning methods
- Using Graph Neural Networks for software metrics
- Applying metaheuristic feature selection
- Scaling federated learning to additional datasets

---

## License

This project is intended for **research and educational purposes**.
