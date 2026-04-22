import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import ISNN1, ISNN2
from numpy_models import ISNN1Numpy, ISNN2Numpy


# -------------------------
# Dataset
# -------------------------
class ISNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------
# Load data
# -------------------------
data = torch.load("problem1_additive.pt")

train_loader = DataLoader(
    ISNNDataset(data["X_train"], data["y_train"]),
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    ISNNDataset(data["X_test"], data["y_test"]),
    batch_size=32,
    shuffle=False
)


# -------------------------
# Helpers
# -------------------------
def split_inputs_torch(X):
    return X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]


def split_inputs_numpy(X):
    return X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Models
# -------------------------
torch_models = {
    "isnn1": ISNN1(1,1,1,1, hidden_size=10, H=2).to(device),
    "isnn2": ISNN2(1,1,1,1, hidden_size=10, H=2).to(device),
}

numpy_models = {
    "isnn1_np": ISNN1Numpy(),
    "isnn2_np": ISNN2Numpy(),
}


# -------------------------
# Optimizers + Loss
# -------------------------
criterion = nn.MSELoss()

optimizers = {
    "isnn1": optim.Adam(torch_models["isnn1"].parameters(), lr=1e-3),
    "isnn2": optim.Adam(torch_models["isnn2"].parameters(), lr=1e-3),
}


# -------------------------
# Logs
# -------------------------
epochs = 200  # (reduce for NumPy fairness)

history = {
    "isnn1_train": [],
    "isnn1_test": [],
    "isnn2_train": [],
    "isnn2_test": [],
    "isnn1_np_test": [],
    "isnn2_np_test": [],
}


print("[train] device =", device)
print("[train] ISNN1 params:", count_parameters(torch_models["isnn1"]))
print("[train] ISNN2 params:", count_parameters(torch_models["isnn2"]))
print("[train] starting...\n")


# =========================================================
# TRAINING LOOP (PyTorch models)
# =========================================================
for epoch in range(epochs):

    # ---------------- ISNN1 ----------------
    model = torch_models["isnn1"]
    model.train()
    train_loss = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        x0, y0, z0, t0 = split_inputs_torch(X)

        pred = model(x0, y0, z0, t0)
        loss = criterion(pred, y)

        optimizers["isnn1"].zero_grad()
        loss.backward()
        optimizers["isnn1"].step()

        train_loss += loss.item()

    history["isnn1_train"].append(train_loss / len(train_loader))

    # eval ISNN1
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            x0, y0, z0, t0 = split_inputs_torch(X)

            pred = model(x0, y0, z0, t0)
            loss = criterion(pred, y)
            test_loss += loss.item()

    history["isnn1_test"].append(test_loss / len(test_loader))


    # ---------------- ISNN2 ----------------
    model = torch_models["isnn2"]
    model.train()
    train_loss = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        x0, y0, z0, t0 = split_inputs_torch(X)

        pred = model(x0, y0, z0, t0)
        loss = criterion(pred, y)

        optimizers["isnn2"].zero_grad()
        loss.backward()
        optimizers["isnn2"].step()

        train_loss += loss.item()

    history["isnn2_train"].append(train_loss / len(train_loader))

    # eval ISNN2
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            x0, y0, z0, t0 = split_inputs_torch(X)

            pred = model(x0, y0, z0, t0)
            loss = criterion(pred, y)
            test_loss += loss.item()

    history["isnn2_test"].append(test_loss / len(test_loader))


    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"ISNN1 train {history['isnn1_train'][-1]:.4f} test {history['isnn1_test'][-1]:.4f} | "
        f"ISNN2 train {history['isnn2_train'][-1]:.4f} test {history['isnn2_test'][-1]:.4f}"
    )


# =========================================================
# NUMPY EVALUATION (after training or parallel runs)
# =========================================================
def eval_numpy(model, loader):
    total = 0

    for X, y in loader:
        X = X.numpy()
        y = y.numpy()

        x0, y0, z0, t0 = split_inputs_numpy(X)

        # IMPORTANT: your design
        # step() already does forward + backward + update
        model.step(x0, y0, z0, t0, y)

        # use predict for evaluation
        pred = model.predict(x0, y0, z0, t0)

        total += ((pred - y) ** 2).mean()

    return total / len(loader)


history["isnn1_np_test"] = eval_numpy(numpy_models["isnn1_np"], test_loader)
history["isnn2_np_test"] = eval_numpy(numpy_models["isnn2_np"], test_loader)


print("\n[Numpy Test Loss]")
print("ISNN1Numpy:", history["isnn1_np_test"])
print("ISNN2Numpy:", history["isnn2_np_test"])


# =========================================================
# PLOTS (PyTorch only curves)
# =========================================================
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(8,5))
plt.plot(epochs_range, history["isnn1_train"], label="ISNN1 Train")
plt.plot(epochs_range, history["isnn2_train"], label="ISNN2 Train")
plt.title("Training Loss (PyTorch)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("train_loss_torch.png", dpi=300)
plt.show()


plt.figure(figsize=(8,5))
plt.plot(epochs_range, history["isnn1_test"], label="ISNN1 Test")
plt.plot(epochs_range, history["isnn2_test"], label="ISNN2 Test")
plt.title("Test Loss (PyTorch)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("test_loss_torch.png", dpi=300)
plt.show()