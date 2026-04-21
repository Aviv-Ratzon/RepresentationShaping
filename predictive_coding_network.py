import math
from utils import get_r_2, calc_PR
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y.astype(int)] = 1.0
    return out


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))


class DeepPredictiveCodingNet:
    """
    Deep predictive coding network trained without backpropagation.

    Core idea:
      1) Initialize neuron states with a feedforward pass.
      2) Run an iterative inference loop that minimizes local prediction errors.
      3) Update each synapse with a local Hebbian-like rule:
             Delta W_l is proportional to e_{l+1}^T times phi(h_l)
         where e_{l+1} is the postsynaptic prediction error and phi(h_l)
         is the presynaptic activity.

    This uses only local quantities during the synaptic update and does not call
    torch.autograd.backward().
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        infer_lr: float = 0.08,
        weight_lr: float = 0.01,
        inference_steps: int = 25,
        activation: str = "tanh",
        device: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least [input_dim, output_dim].")

        self.layer_sizes = list(map(int, layer_sizes))
        self.L = len(self.layer_sizes) - 1
        self.infer_lr = float(infer_lr)
        self.weight_lr = float(weight_lr)
        self.inference_steps = int(inference_steps)
        self.activation = activation
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.W: List[torch.Tensor] = []
        self.b: List[torch.Tensor] = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # Conservative initialization helps predictive-coding inference stay stable.
            scale = math.sqrt(2.0 / (fan_in + fan_out))
            w = torch.randn(fan_out, fan_in, device=self.device) * scale
            b = torch.zeros(fan_out, device=self.device)
            self.W.append(w)
            self.b.append(b)

        self.standardizer: Optional[Standardizer] = None
        self.class_labels_: Optional[np.ndarray] = None
        self.is_classification_: Optional[bool] = None

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "tanh":
            return torch.tanh(x)
        if self.activation == "relu":
            return torch.relu(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _dphi(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "tanh":
            y = torch.tanh(x)
            return 1.0 - y * y
        if self.activation == "relu":
            return (x > 0).to(x.dtype)
        if self.activation == "sigmoid":
            y = torch.sigmoid(x)
            return y * (1.0 - y)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _source_activity(self, state: torch.Tensor, layer_index: int) -> torch.Tensor:
        # Input layer is left linear; hidden layers use the chosen nonlinearity.
        return state if layer_index == 0 else self._phi(state)

    def _source_derivative(self, state: torch.Tensor, layer_index: int) -> torch.Tensor:
        return torch.ones_like(state) if layer_index == 0 else self._dphi(state)

    def _output_activation(self, x: torch.Tensor) -> torch.Tensor:
        # Output layer is always squashed to [0, 1].
        return torch.sigmoid(x)

    def _forward_init(self, x: torch.Tensor) -> List[torch.Tensor]:
        states = [x]
        for l in range(self.L):
            src = self._source_activity(states[l], l)
            pred = src @ self.W[l].T + self.b[l]
            if l == self.L - 1:
                pred = self._output_activation(pred)
            states.append(pred.clone())
        return states

    def _prediction_errors(self, states: List[torch.Tensor]) -> List[Optional[torch.Tensor]]:
        errors: List[Optional[torch.Tensor]] = [None]
        for l in range(self.L):
            src = self._source_activity(states[l], l)
            pred = src @ self.W[l].T + self.b[l]
            if l == self.L - 1:
                pred = self._output_activation(pred)
            errors.append(states[l + 1] - pred)
        return errors

    @torch.no_grad()
    def infer_states(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        states = self._forward_init(x)
        if target is None:
            return states

        for _ in range(self.inference_steps):
            errors = self._prediction_errors(states)
            new_states = [states[0]] + [s.clone() for s in states[1:]]

            # Hidden-state relaxation.
            for l in range(1, self.L):
                top_drive = (errors[l + 1] @ self.W[l]) * self._source_derivative(states[l], l)
                new_states[l] = states[l] + self.infer_lr * (top_drive - errors[l])

            # Output layer: local prediction error + supervised mismatch.
            dloss = states[self.L] - target
            new_states[self.L] = states[self.L] - self.infer_lr * (dloss + errors[self.L])
            states = new_states

        return states

    @torch.no_grad()
    def train_batch(self, x: torch.Tensor, target: torch.Tensor) -> float:
        states = self.infer_states(x, target)
        errors = self._prediction_errors(states)
        batch_size = x.shape[0]

        for l in range(self.L):
            src = self._source_activity(states[l], l)
            err = errors[l + 1]
            dW = err.T @ src / batch_size
            db = err.mean(dim=0)
            self.W[l] += self.weight_lr * dW
            self.b[l] += self.weight_lr * db

        loss = 0.5 * torch.mean((states[self.L] - target) ** 2).item()
        return float(loss)

    def _prepare_data(self, X: ArrayLike, y: ArrayLike, standardize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        X_np = _to_numpy(X).astype(np.float32)
        y_np = _to_numpy(y)

        if standardize:
            self.standardizer = Standardizer.fit(X_np)
            X_np = self.standardizer.transform(X_np)
        else:
            self.standardizer = None

        # Auto-detect classification when y is a 1D integer array.
        if y_np.ndim == 1 and np.issubdtype(y_np.dtype, np.integer):
            self.is_classification_ = True
            self.class_labels_ = np.unique(y_np)
            if not np.array_equal(self.class_labels_, np.arange(len(self.class_labels_))):
                # Remap arbitrary labels to contiguous integers.
                label_to_idx = {label: i for i, label in enumerate(self.class_labels_)}
                y_idx = np.vectorize(label_to_idx.get)(y_np)
            else:
                y_idx = y_np.astype(int)
            y_np = _one_hot(y_idx, num_classes=len(self.class_labels_))
        else:
            self.is_classification_ = False
            self.class_labels_ = None
            y_np = y_np.astype(np.float32)
            if y_np.ndim == 1:
                y_np = y_np[:, None]

        X_t = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        return X_t, y_t

    def _transform_X(self, X: ArrayLike) -> torch.Tensor:
        X_np = _to_numpy(X).astype(np.float32)
        if self.standardizer is not None:
            X_np = self.standardizer.transform(X_np)
        return torch.tensor(X_np, dtype=torch.float32, device=self.device)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        epochs: int = 200,
        batch_size: int = 64,
        shuffle: bool = True,
        standardize: bool = True,
        verbose: bool = True,
    ) -> List[float]:
        X_t, y_t = self._prepare_data(X, y, standardize=standardize)

        if X_t.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Input dimension mismatch: got {X_t.shape[1]}, expected {self.layer_sizes[0]}")
        if y_t.shape[1] != self.layer_sizes[-1]:
            raise ValueError(f"Target dimension mismatch: got {y_t.shape[1]}, expected {self.layer_sizes[-1]}")

        n = X_t.shape[0]
        history: List[float] = []

        for epoch in range(epochs):
            if shuffle:
                perm = torch.randperm(n, device=self.device)
                X_epoch = X_t[perm]
                y_epoch = y_t[perm]
            else:
                X_epoch = X_t
                y_epoch = y_t

            losses = []
            for start in range(0, n, batch_size):
                xb = X_epoch[start : start + batch_size]
                yb = y_epoch[start : start + batch_size]
                losses.append(self.train_batch(xb, yb))

            epoch_loss = float(np.mean(losses))
            history.append(epoch_loss)

            if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0):
                print(f"epoch {epoch + 1:4d}/{epochs} | loss = {epoch_loss:.6f}")

        return history

    @torch.no_grad()
    def forward_states(self, X: ArrayLike) -> List[np.ndarray]:
        X_t = self._transform_X(X)
        states = self.infer_states(X_t, target=None)
        return [_to_numpy(s) for s in states]

    @torch.no_grad()
    def predict(self, X: ArrayLike) -> np.ndarray:
        states = self.forward_states(X)
        out = states[-1]
        if self.is_classification_:
            idx = out.argmax(axis=1)
            if self.class_labels_ is None:
                return idx
            return self.class_labels_[idx]
        return out

    @torch.no_grad()
    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        y_np = _to_numpy(y)
        pred = self.predict(X)
        if self.is_classification_:
            return float(np.mean(pred == y_np))
        pred = np.asarray(pred).reshape(len(y_np), -1)
        target = np.asarray(y_np).reshape(len(y_np), -1)
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-12))

    def get_last_hidden_activations(self, X: ArrayLike) -> np.ndarray:
        states = self.forward_states(X)
        if self.L == 1:
            # No hidden layer exists; fall back to the output layer.
            return states[-1]
        return np.tanh(states[-2]) if self.activation == "tanh" else states[-2]

    def plot_pca_last_layer(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        color: Optional[ArrayLike] = None,
        save_path: Optional[str] = None,
        title: str = "PCA",
        point_size: float = 24.0,
        show: bool = True,
    ) -> None:
        acts = self.get_last_hidden_activations(X).astype(np.float32)
        cmap = 'coolwarm'

        acts = acts - acts.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(acts, full_matrices=False)
        z = u[:, :2] * s[:2]
        r2_score = get_r_2(z[:, :1], y)
        pr_score = calc_PR(acts)

        title += f" --- R^2 = {r2_score:.2f} --- PR = {pr_score:.2f}"

        plt.figure(figsize=(7, 5))
        if y is not None:
            y_np = _to_numpy(y)
            if y_np.ndim == 1:
                scatter = plt.scatter(z[:, 0], z[:, 1], c=y_np, s=point_size, alpha=0.85, cmap=cmap)
                plt.colorbar(scatter, label="label")
            else:
                plt.scatter(z[:, 0], z[:, 1], s=point_size, alpha=0.85, c=color, cmap=cmap)
        else:
            plt.scatter(z[:, 0], z[:, 1], s=point_size, alpha=0.85, c=color, cmap=cmap)

        plt.title(title)
        plt.axis('equal')
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=180, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        return r2_score, pr_score


def train_deep_pcn(
    X: ArrayLike,
    y: ArrayLike,
    L: int,
    hidden_dims: Union[int, Sequence[int]] = 128,
    infer_lr: float = 0.08,
    weight_lr: float = 0.01,
    inference_steps: int = 25,
    activation: str = "tanh",
    epochs: int = 200,
    batch_size: int = 64,
    standardize: bool = True,
    device: Optional[str] = None,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[DeepPredictiveCodingNet, List[float]]:
    """
    Convenience wrapper.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y :
        - classification: integer labels of shape (n_samples,)
        - regression: continuous targets of shape (n_samples, output_dim)
    L : int
        Number of trainable layers (input->...->output). Example:
        L=3 means input -> hidden1 -> hidden2 -> output.
    hidden_dims : int or sequence of int
        If int, all hidden layers share the same width.
        If sequence, its length must be L-1.
    """
    X_np = _to_numpy(X)
    y_np = _to_numpy(y)

    input_dim = int(X_np.shape[1])
    if y_np.ndim == 1 and np.issubdtype(y_np.dtype, np.integer):
        output_dim = int(len(np.unique(y_np)))
    else:
        output_dim = 1 if y_np.ndim == 1 else int(y_np.shape[1])

    if L < 1:
        raise ValueError("L must be >= 1")

    if isinstance(hidden_dims, int):
        hidden = [int(hidden_dims)] * max(0, L - 1)
    else:
        hidden = list(map(int, hidden_dims))
        if len(hidden) != max(0, L - 1):
            raise ValueError(f"hidden_dims must have length {L - 1} when it is a sequence.")

    layer_sizes = [input_dim] + hidden + [output_dim]
    model = DeepPredictiveCodingNet(
        layer_sizes=layer_sizes,
        infer_lr=infer_lr,
        weight_lr=weight_lr,
        inference_steps=inference_steps,
        activation=activation,
        device=device,
        seed=seed,
    )
    history = model.fit(
        X=X,
        y=y,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        standardize=standardize,
        verbose=verbose,
    )
    return model, history


if __name__ == "__main__":
    # Example on a simple classification problem.
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split

    X, y = make_moons(n_samples=1200, noise=0.12, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    model, history = train_deep_pcn(
        X_train,
        y_train,
        L=3,                   # input -> hidden1 -> hidden2 -> output
        hidden_dims=[32, 16],
        infer_lr=0.05,
        weight_lr=0.02,
        inference_steps=15,
        activation="tanh",
        epochs=20,
        batch_size=64,
        seed=0,
        verbose=True,
    )

    print(f"train accuracy: {model.score(X_train, y_train):.3f}")
    print(f"test accuracy : {model.score(X_test, y_test):.3f}")

    model.plot_pca_last_layer(X_test, y_test, save_path="/mnt/data/pcn_last_layer_pca.png", show=False)
