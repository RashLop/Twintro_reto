import numpy as np


class NeuralNetwork:

    def __init__(
        self,
        input_size,
        hidden_sizes=(32, 16, 8),
        learning_rate=0.001,
    ):
        if int(input_size) <= 0:
            raise ValueError("input_size debe ser mayor que 0.")

        if float(learning_rate) <= 0:
            raise ValueError("learning_rate debe ser mayor que 0.")

        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)

        self.input_size = int(input_size)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.learning_rate = float(learning_rate)

        layer_sizes = [self.input_size, *self.hidden_sizes, 1]

        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i + 1]
            weight = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
            bias = np.zeros((1, output_dim), dtype=np.float64)
            self.weights.append(weight.astype(np.float64))
            self.biases.append(bias.astype(np.float64))

        self.activations = []
        self.z_values = []
        self.loss_history = []

    def _prepare_X(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D.")
        if X.shape[1] != self.input_size:
            raise ValueError(
                f"Se esperaban {self.input_size} features, "
                f"pero se recibieron {X.shape[1]}."
            )
        return X

    def forward(self, X):
        X = self._prepare_X(X)
        self.activations = [X]
        self.z_values = []

        activation = X
        for i in range(len(self.weights) - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = np.tanh(z)
            self.z_values.append(z)
            self.activations.append(activation)

        z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        output = z

        self.z_values.append(z)
        self.activations.append(output)
        return output

    def train_batch(self, X, y):
        X = self._prepare_X(X)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        predictions = self.forward(X)
        samples = X.shape[0]

        gradients_w = [None] * len(self.weights)
        gradients_b = [None] * len(self.biases)

        dz = (predictions - y) * (2.0 / samples)

        gradients_w[-1] = self.activations[-2].T @ dz
        gradients_b[-1] = np.sum(dz, axis=0, keepdims=True)

        da = dz @ self.weights[-1].T

        for i in reversed(range(len(self.weights) - 1)):
            activation = self.activations[i + 1]
            dz = da * (1.0 - np.square(activation))
            gradients_w[i] = self.activations[i].T @ dz
            gradients_b[i] = np.sum(dz, axis=0, keepdims=True)

            if i > 0:
                da = dz @ self.weights[i].T

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

        loss = np.mean((predictions - y) ** 2)
        return float(loss)

    def train(self, X, y, epochs=1000, batch_size=256, verbose=True):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D.")

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener el mismo número de muestras.")

        if batch_size <= 0:
            raise ValueError("batch_size debe ser mayor que 0.")

        self.loss_history = []
        samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            batches = 0

            for start in range(0, samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                loss = self.train_batch(X_batch, y_batch)
                epoch_loss += loss
                batches += 1

            epoch_loss /= max(batches, 1)
            self.loss_history.append(epoch_loss)

            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch} | Loss: {epoch_loss:.6f}")

        return self.loss_history

    def predict(self, X):
        prediction = self.forward(X)
        return np.clip(prediction, 0.0, 1.0)

    def save(self, filename):
        data = {
            "input_size": np.array([self.input_size], dtype=np.int64),
            "learning_rate": np.array([self.learning_rate], dtype=np.float64),
        }

        for i, weight in enumerate(self.weights):
            data[f"W{i}"] = weight

        for i, bias in enumerate(self.biases):
            data[f"b{i}"] = bias

        np.savez(filename, **data)

    def load(self, filename):
        data = np.load(filename, allow_pickle=False)

        if "input_size" in data:
            saved_input_size = int(data["input_size"][0])
            if saved_input_size != self.input_size:
                raise ValueError(
                    f"El modelo espera {self.input_size} inputs, "
                    f"pero el archivo contiene {saved_input_size}."
                )

        self.weights = []
        self.biases = []

        i = 0
        while f"W{i}" in data:
            if f"b{i}" not in data:
                raise ValueError(f"Falta el bias de la capa {i}.")
            self.weights.append(data[f"W{i}"].astype(np.float64))
            self.biases.append(data[f"b{i}"].astype(np.float64))
            i += 1

        if not self.weights:
            raise ValueError("El archivo no contiene pesos válidos.")
