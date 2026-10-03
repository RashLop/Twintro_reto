import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network import NeuralNetwork

DATASET_PATH = ROOT / "data" / "training_dataset.json"
MODEL_PATH = ROOT / "matchmaking_model.npz"

HIDDEN_SIZES = (32, 16, 8)
LEARNING_RATE = 0.001
EPOCHS = 1500
BATCH_SIZE = 32
TRAIN_SPLIT = 0.75
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    training_data = json.load(file)

if len(training_data) < 20:
    raise ValueError("Se necesitan al menos 20 interacciones para entrenar el modelo.")

X = np.asarray([sample["features"] for sample in training_data], dtype=np.float64)
y = np.asarray([sample["label"] for sample in training_data], dtype=np.float64).reshape(-1, 1)

if not np.all(np.isfinite(X)):
    raise ValueError("X contiene valores no finitos.")
if not np.all(np.isfinite(y)):
    raise ValueError("y contiene valores no finitos.")
if np.any(y < 0.0) or np.any(y > 1.0):
    raise ValueError("Los labels deben estar entre 0 y 1.")

indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

split = int(len(X) * TRAIN_SPLIT)
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

model = NeuralNetwork(
    input_size=X.shape[1],
    hidden_sizes=HIDDEN_SIZES,
    learning_rate=LEARNING_RATE,
)

print()
print("=" * 70)
print("MATCHMAKING TRAINING (REGRESIÓN)")
print("=" * 70)
print(f"Total:       {len(X)}")
print(f"Training:    {len(X_train)}")
print(f"Testing:     {len(X_test)}")
print(f"Features:    {X.shape[1]}")
print(f"Architecture: {X.shape[1]} -> {HIDDEN_SIZES[0]} -> {HIDDEN_SIZES[1]} -> {HIDDEN_SIZES[2]} -> 1")

best_loss = float("inf")
best_weights = None
best_biases = None
patience = 0

for epoch in range(EPOCHS):
    model.train(X_train, y_train, epochs=1, batch_size=BATCH_SIZE, verbose=False)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_loss = np.mean((train_pred - y_train) ** 2)
    test_loss = np.mean((test_pred - y_test) ** 2)
    test_mae = np.mean(np.abs(test_pred - y_test))

    if test_loss < best_loss:
        best_loss = float(test_loss)
        best_weights = [w.copy() for w in model.weights]
        best_biases = [b.copy() for b in model.biases]
        patience = 0
    else:
        patience += 1

    if epoch % 250 == 0:
        print(f"Epoch {epoch:4d} | MSE={test_loss:.6f} | MAE={test_mae:.4f}")

    if patience >= 300:
        break

if best_weights is not None:
    model.weights = best_weights
    model.biases = best_biases

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
print("\nEvaluación final")
print(f"Train MSE: {np.mean((train_pred - y_train) ** 2):.6f}")
print(f"Test MSE:  {np.mean((test_pred - y_test) ** 2):.6f}")
print(f"Test MAE:  {np.mean(np.abs(test_pred - y_test)):.4f}")

ranked = np.argsort(test_pred[:, 0])[::-1]
if len(ranked) >= 10:
    good_mean = float(test_pred[ranked[:max(5, len(ranked) // 10)], 0].mean())
    bad_mean = float(test_pred[ranked[-max(5, len(ranked) // 10):], 0].mean())
    print(f"Good pairs mean: {good_mean:.4f}")
    print(f"Bad pairs mean:  {bad_mean:.4f}")
    print(f"Gap:             {good_mean - bad_mean:.4f}")
    if good_mean <= bad_mean + 0.05:
        raise ValueError("El modelo no diferencia claramente entre pares buenos y malos.")

model.save(MODEL_PATH)
print(f"Modelo guardado en: {MODEL_PATH}")
print("=" * 70)
