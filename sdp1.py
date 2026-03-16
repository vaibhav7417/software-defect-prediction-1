import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np

# Load and preprocess datasets

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Apply ADASYN for class balancing
    adasyn = ADASYN()
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    # Apply polynomial feature expansion
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_resampled = poly.fit_transform(X_resampled)

    # Normalize features
    scaler = StandardScaler()
    X_resampled = scaler.fit_transform(X_resampled)

    return TensorDataset(torch.tensor(X_resampled, dtype=torch.float32), torch.tensor(y_resampled, dtype=torch.long))

# Load CM1, KC1, PC1 datasets
datasets = {
    "CM1": load_dataset("/content/cm1.csv"),
    "KC1": load_dataset("/content/kc1.csv"),
    "PC1": load_dataset("/content/pc1.csv")
}

# Define Neural Network Model
class FLModel(nn.Module):
    def __init__(self, input_dim):
        super(FLModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.03)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2)  # Binary classification

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        return self.fc6(x)

# Local Training with FedProx Regularization
def train_local(model, dataset, global_weights, mu=0.0025, epochs=18, lr=7e-5):
    labels = torch.tensor([y for _, y in dataset], dtype=torch.long)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels.numpy()), y=labels.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    sample_weights = [class_weights[y.item()] for _, y in dataset]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    loader = DataLoader(dataset, batch_size=32, sampler=sampler, drop_last=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # FedProx Regularization
            fed_prox_reg = sum(torch.norm(model.state_dict()[key] - global_weights[key].float()) ** 2 for key in global_weights.keys())
            loss += (mu / 2) * fed_prox_reg

            loss.backward()
            optimizer.step()
        scheduler.step()

    return model.state_dict()

# Federated Averaging Function
def federated_averaging(global_model, client_weights, client_data_sizes):
    total_samples = sum(client_data_sizes)
    new_weights = global_model.state_dict()

    for key in new_weights.keys():
        new_weights[key] = sum(client_weights[i][key] * (client_data_sizes[i] / total_samples)
                               for i in range(len(client_weights)))

    global_model.load_state_dict(new_weights)
    return global_model

# Evaluation Function
def evaluate_model(model, dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=1)
    recall = recall_score(all_labels, all_preds, zero_division=1)
    f1 = f1_score(all_labels, all_preds, zero_division=1)

    return accuracy, precision, recall, f1

# Initialize Global Model
input_dim = datasets["CM1"][0][0].shape[0]
global_model = FLModel(input_dim)

# Simulated Federated Learning Process
rounds = 18
global_weights = global_model.state_dict()

for r in range(rounds):
    print(f"--- Round {r+1} ---")

    client_weights = []
    client_sizes = []

    for client, data in datasets.items():
        local_model = FLModel(input_dim)
        local_model.load_state_dict(global_weights)  # Start from global model

        updated_weights = train_local(local_model, data, global_weights)
        client_weights.append(updated_weights)
        client_sizes.append(len(data))

    # Aggregate using FedAvg
    global_model = federated_averaging(global_model, client_weights, client_sizes)
    global_weights = global_model.state_dict()

# Evaluate Final Model
for client, data in datasets.items():
    accuracy, precision, recall, f1 = evaluate_model(global_model, data)
    print(f"{client} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

print("Federated Learning Training & Evaluation Complete!")
