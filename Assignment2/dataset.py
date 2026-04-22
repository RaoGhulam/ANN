import numpy as np
from scipy.stats import qmc
import torch


# ===============================
# Save to PyTorch format
# ===============================
def save_torch_dataset(filename, X_train, y_train, X_test, y_test):
    torch.save({
        "X_train": torch.from_numpy(X_train).float(),
        "y_train": torch.from_numpy(y_train).float().view(-1, 1),
        "X_test": torch.from_numpy(X_test).float(),
        "y_test": torch.from_numpy(y_test).float().view(-1, 1),
    }, filename)


# ===============================
# Latin Hypercube Sampling
# ===============================
def lhs_sampling(n_samples, dim, bounds):
    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(n=n_samples)
    return qmc.scale(sample, bounds[:, 0], bounds[:, 1])


# ===============================
# Problem 1: Additive Split
# ===============================
def f_additive(x, y, t, z):
    return (
        np.exp(-0.5 * x)
        + np.log(1 + np.exp(0.4 * y))
        + np.tanh(t)
        + np.sin(z)
        - 0.4
    )


def generate_problem1():
    dim = 4

    train_bounds = np.array([[0, 4], [0, 4], [0, 4], [0, 4]])
    test_bounds  = np.array([[0, 6], [0, 6], [0, 6], [0, 6]])

    X_train = lhs_sampling(500, dim, train_bounds)
    X_test  = lhs_sampling(5000, dim, test_bounds)

    y_train = f_additive(*X_train.T)
    y_test  = f_additive(*X_test.T)

    return X_train, y_train, X_test, y_test


# ===============================
# Problem 2: Multiplicative Split
# (fixed variable order: x, y, t, z)
# ===============================
def f_multiplicative(x, y, t, z):
    f_x = np.exp(-0.3 * x)
    f_y = (0.15 * y) ** 2
    f_t = np.tanh(0.3 * t)
    f_z = 0.2 * np.sin(0.5 * z + 2) + 0.5
    return f_x * f_y * f_z * f_t


def generate_problem2():
    dim = 4

    train_bounds = np.array([[0, 4], [0, 4], [0, 4], [0, 4]])
    test_bounds  = np.array([[0, 10], [0, 10], [0, 10], [0, 10]])

    X_train = lhs_sampling(500, dim, train_bounds)
    X_test  = lhs_sampling(5000, dim, test_bounds)

    y_train = f_multiplicative(*X_train.T)
    y_test  = f_multiplicative(*X_test.T)

    return X_train, y_train, X_test, y_test


# ===============================
# Main execution
# ===============================
if __name__ == "__main__":

    # ======================
    # Problem 1 (Additive)
    # ======================
    Xtr1, ytr1, Xte1, yte1 = generate_problem1()

    save_torch_dataset(
        "problem1_additive.pt",
        Xtr1, ytr1, Xte1, yte1
    )

    print("Saved Problem 1 → problem1_additive.pt")

    # ======================
    # Problem 2 (Multiplicative)
    # ======================
    # Xtr2, ytr2, Xte2, yte2 = generate_problem2()

    # save_torch_dataset(
    #     "problem2_multiplicative.pt",
    #     Xtr2, ytr2, Xte2, yte2
    # )

    # print("Saved Problem 2 → problem2_multiplicative.pt")