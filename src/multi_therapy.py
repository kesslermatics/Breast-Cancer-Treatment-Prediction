import kagglehub
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset

# Download the METABRIC dataset from KaggleHub
path = kagglehub.dataset_download("raghadalharbi/breast-cancer-gene-expression-profiles-metabric")
print("Path to dataset files:", path)

# Load dataset
df = pd.read_csv(os.path.join(path, "METABRIC_RNA_Mutation.csv"))
device = torch.device("cpu")

# Selected features for prediction
selected_features = [
    "age_at_diagnosis", "tumor_size", "tumor_stage", "lymph_nodes_examined_positive",
    "nottingham_prognostic_index", "cellularity", "neoplasm_histologic_grade",
    "inferred_menopausal_state", "er_status_measured_by_ihc", "pr_status", "her2_status"
]

# Create a simplified therapy class label:
# 0 = Chemotherapy only, 1 = Radiotherapy only, 2 = Hormone therapy only
df["therapy_class_simple"] = -1
df.loc[(df["chemotherapy"] == 1) & (df["hormone_therapy"] == 0) & (df["radio_therapy"] == 0), "therapy_class_simple"] = 0
df.loc[(df["chemotherapy"] == 0) & (df["hormone_therapy"] == 0) & (df["radio_therapy"] == 1), "therapy_class_simple"] = 1
df.loc[(df["chemotherapy"] == 0) & (df["hormone_therapy"] == 1) & (df["radio_therapy"] == 0), "therapy_class_simple"] = 2

# Keep only valid rows
df = df[df["therapy_class_simple"] != -1].copy()
df = df.dropna(subset=selected_features + ["therapy_class_simple"])

# Define features and target
X = df[selected_features].copy()
y = df["therapy_class_simple"].astype(int)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
X[categorical_cols] = X[categorical_cols].astype(str)
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Preprocessing pipeline with scaling and one-hot encoding
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
])
X_processed = preprocessor.fit_transform(X)

input_dim = X_processed.shape[1]
num_classes = len(np.unique(y))

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.long).to(device)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, stratify=y, random_state=42
)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# Define the neural network model for multiclass classification
class MulticlassTherapyModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# Initialize model, loss function, and optimizer
model = MulticlassTherapyModel(input_dim, num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Setup for early stopping
best_val_loss = float("inf")
patience = 10
counter = 0
best_model_state = None
train_losses, val_losses = [], []

# Training loop
for epoch in range(200):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        preds = model(xb)
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_preds = model(X_test)
        val_loss = loss_fn(val_preds, y_test)
        val_losses.append(val_loss.item())

    print(f"Epoch {epoch+1}/200 | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    # Check early stopping condition
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_model_state = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Plot and save learning curve
os.makedirs("../outputs/figures", exist_ok=True)
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("CrossEntropy Loss")
plt.title("Learning Curve (Therapy: Chemo vs Radio vs Hormon)")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/nn_learning_curve_therapy_simple.png")

# Model evaluation
model.eval()
with torch.no_grad():
    logits = model(X_test)
    y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    y_true = y_test.cpu().numpy()

# Print classification report
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)

# Plot and save confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Multiclass Therapy Prediction")
plt.tight_layout()
os.makedirs("../outputs/figures", exist_ok=True)
plt.savefig("../outputs/figures/confusion_matrix_multiclass_therapy.png")
