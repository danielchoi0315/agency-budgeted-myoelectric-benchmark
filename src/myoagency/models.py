from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None


ArrayInput = np.ndarray | dict[str, np.ndarray]
ModelRole = Literal["either", "user", "assist"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    input_kind: Literal["tabular", "sequence", "hybrid"] = "tabular"
    role: ModelRole = "either"
    required_sequence_keys: tuple[str, ...] = ()


def shallow_model_specs(random_seed: int = 20260415) -> list[ModelSpec]:
    specs: list[ModelSpec] = [
        ModelSpec("lda", Pipeline([("scale", StandardScaler()), ("clf", LinearDiscriminantAnalysis())])),
        ModelSpec(
            "logistic",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "extra_trees",
            Pipeline(
                [
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=512,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=random_seed,
                        ),
                    )
                ]
            ),
        ),
        ModelSpec(
            "random_forest",
            Pipeline(
                [
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=512,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=random_seed,
                        ),
                    )
                ]
            ),
        ),
        ModelSpec(
            "svm_rbf",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("clf", SVC(C=3.0, gamma="scale", probability=True, class_weight="balanced", random_state=random_seed)),
                ]
            ),
        ),
        ModelSpec(
            "assist_temporal_forest",
            AssistTemporalForestClassifier(
                n_estimators=512,
                n_chunks=4,
                random_state=random_seed,
            ),
            input_kind="hybrid",
            role="assist",
            required_sequence_keys=(
                "assist_emg",
                "assist_time_mask",
                "assist_gaze",
                "assist_gaze_mask",
                "assist_pupil",
                "assist_pupil_mask",
                "assist_acc",
                "assist_acc_mask",
                "assist_gyr",
                "assist_gyr_mask",
                "assist_context",
                "assist_context_mask",
            ),
        ),
    ]
    if torch is not None:
        specs.extend(
            [
                ModelSpec(
                    "torch_mlp",
                    Pipeline(
                        [
                            ("scale", StandardScaler()),
                            (
                                "clf",
                                TorchMLPClassifier(
                                    hidden_dims=(512, 256),
                                    dropout=0.10,
                                    batch_size=8192,
                                    max_epochs=36,
                                    lr=3e-3,
                                    weight_decay=2e-4,
                                    val_fraction=0.1,
                                    patience=5,
                                    random_state=random_seed,
                                ),
                            ),
                        ]
                    ),
                ),
                ModelSpec(
                    "temporal_cnn",
                    TemporalSequenceClassifier(
                        payload_key="user_emg",
                        mask_key="user_time_mask",
                        hidden_dim=128,
                        kernel_size=5,
                        batch_size=256,
                        max_epochs=24,
                        lr=2e-3,
                        weight_decay=2e-4,
                        dropout=0.15,
                        val_fraction=0.1,
                        patience=5,
                        random_state=random_seed,
                    ),
                    input_kind="sequence",
                    role="user",
                    required_sequence_keys=("user_emg", "user_time_mask"),
                ),
                ModelSpec(
                    "hybrid_temporal_mlp",
                    HybridTemporalTabularClassifier(
                        payload_key="user_emg",
                        mask_key="user_time_mask",
                        tabular_key="tabular",
                        sequence_hidden_dim=160,
                        tabular_hidden_dim=256,
                        fusion_hidden_dim=256,
                        kernel_size=5,
                        batch_size=256,
                        max_epochs=26,
                        lr=1.6e-3,
                        weight_decay=2e-4,
                        dropout=0.15,
                        val_fraction=0.1,
                        patience=6,
                        random_state=random_seed,
                    ),
                    input_kind="hybrid",
                    role="user",
                    required_sequence_keys=("user_emg", "user_time_mask"),
                ),
                ModelSpec(
                    "pactformer",
                    PACTFormerClassifier(
                        token_dim=160,
                        num_heads=4,
                        num_layers=2,
                        branch_hidden_dim=96,
                        batch_size=128,
                        max_epochs=28,
                        lr=7e-4,
                        weight_decay=2e-4,
                        dropout=0.15,
                        modality_dropout=0.08,
                        val_fraction=0.1,
                        patience=6,
                        random_state=random_seed,
                    ),
                    input_kind="sequence",
                    role="assist",
                    required_sequence_keys=(
                        "assist_emg",
                        "assist_time_mask",
                        "assist_gaze",
                        "assist_gaze_mask",
                        "assist_pupil",
                        "assist_pupil_mask",
                        "assist_acc",
                        "assist_acc_mask",
                        "assist_gyr",
                        "assist_gyr_mask",
                        "assist_context",
                        "assist_context_mask",
                    ),
                ),
                ModelSpec(
                    "assist_hybrid_fusion",
                    AssistHybridFusionClassifier(
                        token_dim=160,
                        num_heads=4,
                        num_layers=2,
                        branch_hidden_dim=96,
                        tabular_hidden_dim=256,
                        modality_dropout=0.05,
                        batch_size=128,
                        max_epochs=28,
                        lr=9e-4,
                        weight_decay=2e-4,
                        dropout=0.15,
                        val_fraction=0.1,
                        patience=6,
                        random_state=random_seed,
                    ),
                    input_kind="hybrid",
                    role="assist",
                    required_sequence_keys=(
                        "assist_emg",
                        "assist_time_mask",
                        "assist_gaze",
                        "assist_gaze_mask",
                        "assist_pupil",
                        "assist_pupil_mask",
                        "assist_acc",
                        "assist_acc_mask",
                        "assist_gyr",
                        "assist_gyr_mask",
                        "assist_context",
                        "assist_context_mask",
                    ),
                ),
                ModelSpec(
                    "assist_stacked_fusion",
                    AssistStackedFusionClassifier(
                        alpha_grid=(0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0),
                        val_fraction=0.1,
                        n_estimators=512,
                        token_dim=160,
                        num_heads=4,
                        num_layers=2,
                        branch_hidden_dim=96,
                        tabular_hidden_dim=256,
                        modality_dropout=0.05,
                        batch_size=128,
                        max_epochs=28,
                        lr=9e-4,
                        weight_decay=2e-4,
                        dropout=0.15,
                        patience=6,
                        random_state=random_seed,
                    ),
                    input_kind="hybrid",
                    role="assist",
                    required_sequence_keys=(
                        "assist_emg",
                        "assist_time_mask",
                        "assist_gaze",
                        "assist_gaze_mask",
                        "assist_pupil",
                        "assist_pupil_mask",
                        "assist_acc",
                        "assist_acc_mask",
                        "assist_gyr",
                        "assist_gyr_mask",
                        "assist_context",
                        "assist_context_mask",
                    ),
                ),
            ]
        )
    return specs


def input_row_count(x: ArrayInput) -> int:
    if isinstance(x, dict):
        if not x:
            raise ValueError("sequence input payload is empty")
        counts = {int(np.asarray(value).shape[0]) for value in x.values()}
        if len(counts) != 1:
            raise ValueError(f"sequence payload row-count mismatch: {sorted(counts)}")
        return counts.pop()
    return int(np.asarray(x).shape[0])


def slice_input_rows(x: ArrayInput, indices: np.ndarray) -> ArrayInput:
    if isinstance(x, dict):
        return {name: np.asarray(value)[indices] for name, value in x.items()}
    return np.asarray(x)[indices]


def fit_predict_proba(
    estimator: object,
    x_train: ArrayInput,
    y_train: np.ndarray,
    x_test: ArrayInput,
    class_labels: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    estimator.fit(x_train, y_train)
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("estimator must expose predict_proba")
    probs = np.asarray(estimator.predict_proba(x_test), dtype=float)
    classes = getattr(estimator, "classes_", None)
    if classes is None and hasattr(estimator, "named_steps"):
        for step in reversed(list(estimator.named_steps.values())):
            step_classes = getattr(step, "classes_", None)
            if step_classes is not None:
                classes = step_classes
                break
    if class_labels is None:
        if classes is None:
            return probs
        class_labels = np.arange(int(np.max(classes)) + 1)
    class_labels = np.asarray(class_labels, dtype=int)
    classes = np.asarray(classes, dtype=int)
    missing = set(classes) - set(class_labels)
    if missing:
        raise ValueError(f"estimator emitted classes outside declared vocabulary: {sorted(missing)}")
    full = np.zeros((input_row_count(x_test), class_labels.shape[0]), dtype=float)
    class_to_col = {label: idx for idx, label in enumerate(class_labels)}
    for src_col, label in enumerate(classes):
        full[:, class_to_col[int(label)]] = probs[:, src_col]
    return full


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.15,
        batch_size: int = 4096,
        max_epochs: int = 40,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        val_fraction: float = 0.1,
        patience: int = 5,
        device: str = "auto",
        random_state: int = 20260415,
        use_amp: bool = True,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.use_amp = use_amp

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchMLPClassifier":
        if torch is None or nn is None:
            raise ImportError("torch is required for TorchMLPClassifier")
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_features_in_ = int(x.shape[1])
        if self.classes_.shape[0] == 1:
            self._single_class = int(self.classes_[0])
            self._network = None
            return self
        self._single_class = None
        device = self._resolve_device()
        self.device_ = str(device)
        rng = np.random.default_rng(self.random_state)
        train_index, val_index = _train_val_split(y_encoded, self.val_fraction, rng)
        x_train = x[train_index]
        y_train = y_encoded[train_index]
        x_val = x[val_index] if val_index.size else None
        y_val = y_encoded[val_index] if val_index.size else None
        _configure_torch(device, self.random_state)
        network = self._build_network(input_dim=x.shape[1], n_classes=self.classes_.shape[0]).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        class_weights = _balanced_class_weights(y_train, self.classes_.shape[0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        use_amp = bool(device.type == "cuda" and self.use_amp)
        scaler = _build_grad_scaler(use_amp)
        best_state = None
        best_metric = float("inf")
        patience_left = int(self.patience)
        for _ in range(int(self.max_epochs)):
            network.train()
            for batch_x, batch_y in _iter_numpy_batches(x_train, y_train, int(self.batch_size), rng):
                batch_x_tensor = torch.from_numpy(batch_x).to(device, non_blocking=True)
                batch_y_tensor = torch.from_numpy(batch_y).to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = network(batch_x_tensor)
                    loss = criterion(logits, batch_y_tensor)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            metric = _validation_loss_numpy(network, criterion, x_val, y_val, int(self.batch_size), device, use_amp)
            if metric + 1e-6 < best_metric:
                best_metric = metric
                patience_left = int(self.patience)
                best_state = {key: value.detach().cpu() for key, value in network.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is None:
            best_state = {key: value.detach().cpu() for key, value in network.state_dict().items()}
        network.load_state_dict(best_state)
        self._network = network.to("cpu")
        self._network.eval()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self._single_class is not None:
            probs = np.zeros((x.shape[0], 1), dtype=np.float32)
            probs[:, 0] = 1.0
            return probs
        assert self._network is not None
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, x.shape[0], int(self.batch_size)):
                stop = min(start + int(self.batch_size), x.shape[0])
                batch = torch.from_numpy(x[start:stop])
                logits = self._network(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                outputs.append(probs)
        return np.vstack(outputs)

    def _build_network(self, input_dim: int, n_classes: int):
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for hidden_dim in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(self.dropout)),
                ]
            )
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(n_classes)))
        return nn.Sequential(*layers)

    def _resolve_device(self):
        if torch is None:
            raise ImportError("torch is required for TorchMLPClassifier")
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TorchSequenceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        batch_size: int = 256,
        max_epochs: int = 24,
        lr: float = 2e-3,
        weight_decay: float = 2e-4,
        dropout: float = 0.15,
        val_fraction: float = 0.1,
        patience: int = 5,
        grad_clip: float = 1.0,
        device: str = "auto",
        random_state: int = 20260415,
        use_amp: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.val_fraction = val_fraction
        self.patience = patience
        self.grad_clip = grad_clip
        self.device = device
        self.random_state = random_state
        self.use_amp = use_amp

    def fit(self, x: ArrayInput, y: np.ndarray):
        if torch is None or nn is None:
            raise ImportError("torch is required for sequence classifiers")
        y = np.asarray(y, dtype=int)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        if self.classes_.shape[0] == 1:
            self._single_class = int(self.classes_[0])
            self._network = None
            return self
        self._single_class = None
        x_checked = _validate_sequence_input(x)
        device = self._resolve_device()
        self.device_ = str(device)
        rng = np.random.default_rng(self.random_state)
        train_index, val_index = _train_val_split(y_encoded, self.val_fraction, rng)
        x_train = slice_input_rows(x_checked, train_index)
        y_train = y_encoded[train_index]
        x_val = slice_input_rows(x_checked, val_index) if val_index.size else None
        y_val = y_encoded[val_index] if val_index.size else None
        self._normalizers = _fit_input_normalizers(x_train)
        _configure_torch(device, self.random_state)
        network = self._build_network(x_train, int(self.classes_.shape[0])).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=_balanced_class_weights(y_train, self.classes_.shape[0]).to(device))
        use_amp = bool(device.type == "cuda" and self.use_amp)
        scaler = _build_grad_scaler(use_amp)
        best_state = None
        best_metric = float("inf")
        patience_left = int(self.patience)
        for _ in range(int(self.max_epochs)):
            network.train()
            for batch_index in _iter_index_batches(len(y_train), int(self.batch_size), rng):
                batch_x = _prepare_torch_batch(
                    slice_input_rows(x_train, batch_index),
                    device=device,
                    normalizers=self._normalizers,
                )
                batch_y = torch.from_numpy(y_train[batch_index]).to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits, aux_loss = _unpack_network_output(network(batch_x))
                    loss = criterion(logits, batch_y) + aux_loss
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(network.parameters(), float(self.grad_clip))
                scaler.step(optimizer)
                scaler.update()
            metric = _validation_loss_sequence(
                network,
                criterion,
                x_val,
                y_val,
                batch_size=int(self.batch_size),
                normalizers=self._normalizers,
                device=device,
                use_amp=use_amp,
            )
            if metric + 1e-6 < best_metric:
                best_metric = metric
                patience_left = int(self.patience)
                best_state = {key: value.detach().cpu() for key, value in network.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is None:
            best_state = {key: value.detach().cpu() for key, value in network.state_dict().items()}
        network.load_state_dict(best_state)
        self._network = network.to("cpu")
        self._network.eval()
        return self

    def predict_proba(self, x: ArrayInput) -> np.ndarray:
        x_checked = _validate_sequence_input(x)
        n_rows = input_row_count(x_checked)
        if self._single_class is not None:
            probs = np.zeros((n_rows, 1), dtype=np.float32)
            probs[:, 0] = 1.0
            return probs
        outputs: list[np.ndarray] = []
        assert self._network is not None
        with torch.inference_mode():
            for batch_index in _iter_ordered_index_batches(n_rows, int(self.batch_size)):
                batch_x = _prepare_torch_batch(
                    slice_input_rows(x_checked, batch_index),
                    device=torch.device("cpu"),
                    normalizers=self._normalizers,
                )
                logits, _ = _unpack_network_output(self._network(batch_x))
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(outputs)

    def _resolve_device(self):
        if torch is None:
            raise ImportError("torch is required for sequence classifiers")
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_network(self, x_train: ArrayInput, n_classes: int):
        raise NotImplementedError


class TemporalSequenceClassifier(_TorchSequenceClassifier):
    def __init__(
        self,
        payload_key: str,
        mask_key: str | None = None,
        hidden_dim: int = 128,
        kernel_size: int = 5,
        batch_size: int = 256,
        max_epochs: int = 24,
        lr: float = 2e-3,
        weight_decay: float = 2e-4,
        dropout: float = 0.15,
        val_fraction: float = 0.1,
        patience: int = 5,
        grad_clip: float = 1.0,
        device: str = "auto",
        random_state: int = 20260415,
        use_amp: bool = True,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            val_fraction=val_fraction,
            patience=patience,
            grad_clip=grad_clip,
            device=device,
            random_state=random_state,
            use_amp=use_amp,
        )
        self.payload_key = payload_key
        self.mask_key = mask_key
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

    def _build_network(self, x_train: ArrayInput, n_classes: int):
        assert isinstance(x_train, dict)
        input_dim = int(np.asarray(x_train[self.payload_key]).shape[-1])
        return _TemporalSequenceNetwork(
            payload_key=self.payload_key,
            mask_key=self.mask_key,
            input_dim=input_dim,
            hidden_dim=int(self.hidden_dim),
            n_classes=int(n_classes),
            dropout=float(self.dropout),
            kernel_size=int(self.kernel_size),
        )


class PACTFormerClassifier(_TorchSequenceClassifier):
    def __init__(
        self,
        token_dim: int = 160,
        num_heads: int = 4,
        num_layers: int = 2,
        branch_hidden_dim: int = 96,
        modality_dropout: float = 0.08,
        batch_size: int = 128,
        max_epochs: int = 28,
        lr: float = 1.5e-3,
        weight_decay: float = 2e-4,
        dropout: float = 0.15,
        val_fraction: float = 0.1,
        patience: int = 6,
        grad_clip: float = 1.0,
        device: str = "auto",
        random_state: int = 20260415,
        use_amp: bool = True,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            val_fraction=val_fraction,
            patience=patience,
            grad_clip=grad_clip,
            device=device,
            random_state=random_state,
            use_amp=use_amp,
        )
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.branch_hidden_dim = branch_hidden_dim
        self.modality_dropout = modality_dropout

    def _build_network(self, x_train: ArrayInput, n_classes: int):
        assert isinstance(x_train, dict)
        modality_dims = {
            "assist_emg": int(np.asarray(x_train["assist_emg"]).shape[-1]),
            "assist_gaze": int(np.asarray(x_train["assist_gaze"]).shape[-1]),
            "assist_pupil": int(np.asarray(x_train["assist_pupil"]).shape[-1]),
            "assist_acc": int(np.asarray(x_train["assist_acc"]).shape[-1]),
            "assist_gyr": int(np.asarray(x_train["assist_gyr"]).shape[-1]),
            "assist_context": int(np.asarray(x_train["assist_context"]).shape[-1]),
        }
        return _PACTFormerNetwork(
            modality_dims=modality_dims,
            token_dim=int(self.token_dim),
            num_heads=int(self.num_heads),
            num_layers=int(self.num_layers),
            branch_hidden_dim=int(self.branch_hidden_dim),
            n_classes=int(n_classes),
            dropout=float(self.dropout),
            modality_dropout=float(self.modality_dropout),
        )


class AssistHybridFusionClassifier(_TorchSequenceClassifier):
    REQUIRED_SEQUENCE_KEYS = (
        "assist_emg",
        "assist_time_mask",
        "assist_gaze",
        "assist_gaze_mask",
        "assist_pupil",
        "assist_pupil_mask",
        "assist_acc",
        "assist_acc_mask",
        "assist_gyr",
        "assist_gyr_mask",
        "assist_context",
        "assist_context_mask",
    )

    def __init__(
        self,
        tabular_key: str = "tabular",
        token_dim: int = 160,
        num_heads: int = 4,
        num_layers: int = 2,
        branch_hidden_dim: int = 96,
        tabular_hidden_dim: int = 256,
        modality_dropout: float = 0.05,
        batch_size: int = 128,
        max_epochs: int = 28,
        lr: float = 9e-4,
        weight_decay: float = 2e-4,
        dropout: float = 0.15,
        val_fraction: float = 0.1,
        patience: int = 6,
        grad_clip: float = 1.0,
        device: str = "auto",
        random_state: int = 20260415,
        use_amp: bool = True,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            val_fraction=val_fraction,
            patience=patience,
            grad_clip=grad_clip,
            device=device,
            random_state=random_state,
            use_amp=use_amp,
        )
        self.tabular_key = tabular_key
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.branch_hidden_dim = branch_hidden_dim
        self.tabular_hidden_dim = tabular_hidden_dim
        self.modality_dropout = modality_dropout

    def prepare_benchmark_input(
        self,
        *,
        tabular: np.ndarray,
        sequence: dict[str, np.ndarray] | None,
        **_: object,
    ) -> dict[str, np.ndarray]:
        if sequence is None:
            raise ValueError("assist hybrid fusion classifier requires aligned assist sequence payloads")
        payload = {name: np.asarray(value) for name, value in sequence.items()}
        missing = [key for key in self.REQUIRED_SEQUENCE_KEYS if key not in payload]
        if missing:
            raise ValueError(f"assist hybrid fusion payload is missing required keys: {missing}")
        payload[self.tabular_key] = np.asarray(tabular, dtype=np.float32)
        return payload

    def _build_network(self, x_train: ArrayInput, n_classes: int):
        assert isinstance(x_train, dict)
        modality_dims = {
            "assist_emg": int(np.asarray(x_train["assist_emg"]).shape[-1]),
            "assist_gaze": int(np.asarray(x_train["assist_gaze"]).shape[-1]),
            "assist_pupil": int(np.asarray(x_train["assist_pupil"]).shape[-1]),
            "assist_acc": int(np.asarray(x_train["assist_acc"]).shape[-1]),
            "assist_gyr": int(np.asarray(x_train["assist_gyr"]).shape[-1]),
            "assist_context": int(np.asarray(x_train["assist_context"]).shape[-1]),
        }
        tabular = np.asarray(x_train[self.tabular_key])
        tabular_dim = int(tabular.shape[-1]) if tabular.ndim >= 2 else 1
        return _AssistHybridFusionNetwork(
            modality_dims=modality_dims,
            tabular_key=self.tabular_key,
            tabular_dim=tabular_dim,
            token_dim=int(self.token_dim),
            num_heads=int(self.num_heads),
            num_layers=int(self.num_layers),
            branch_hidden_dim=int(self.branch_hidden_dim),
            tabular_hidden_dim=int(self.tabular_hidden_dim),
            n_classes=int(n_classes),
            dropout=float(self.dropout),
            modality_dropout=float(self.modality_dropout),
        )


class AssistStackedFusionClassifier(BaseEstimator, ClassifierMixin):
    REQUIRED_SEQUENCE_KEYS = AssistHybridFusionClassifier.REQUIRED_SEQUENCE_KEYS

    def __init__(
        self,
        tabular_key: str = "tabular",
        alpha_grid: tuple[float, ...] = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0),
        val_fraction: float = 0.1,
        n_estimators: int = 512,
        token_dim: int = 160,
        num_heads: int = 4,
        num_layers: int = 2,
        branch_hidden_dim: int = 96,
        tabular_hidden_dim: int = 256,
        modality_dropout: float = 0.05,
        batch_size: int = 128,
        max_epochs: int = 28,
        lr: float = 9e-4,
        weight_decay: float = 2e-4,
        dropout: float = 0.15,
        patience: int = 6,
        device: str = "auto",
        random_state: int = 20260415,
        use_amp: bool = True,
    ) -> None:
        self.tabular_key = tabular_key
        self.alpha_grid = alpha_grid
        self.val_fraction = val_fraction
        self.n_estimators = n_estimators
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.branch_hidden_dim = branch_hidden_dim
        self.tabular_hidden_dim = tabular_hidden_dim
        self.modality_dropout = modality_dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.use_amp = use_amp

    def prepare_benchmark_input(
        self,
        *,
        tabular: np.ndarray,
        sequence: dict[str, np.ndarray] | None,
        **_: object,
    ) -> dict[str, np.ndarray]:
        if sequence is None:
            raise ValueError("assist stacked fusion classifier requires aligned assist sequence payloads")
        payload = {name: np.asarray(value) for name, value in sequence.items()}
        missing = [key for key in self.REQUIRED_SEQUENCE_KEYS if key not in payload]
        if missing:
            raise ValueError(f"assist stacked fusion payload is missing required keys: {missing}")
        payload[self.tabular_key] = np.asarray(tabular, dtype=np.float32)
        return payload

    def fit(self, x: ArrayInput, y: np.ndarray) -> "AssistStackedFusionClassifier":
        x_checked = _validate_sequence_input(x)
        if not isinstance(x_checked, dict):
            raise ValueError("assist stacked fusion classifier expects a hybrid input mapping")
        y = np.asarray(y, dtype=int)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        if self.classes_.shape[0] == 1:
            self._single_class = int(self.classes_[0])
            self._tabular_model = None
            self._deep_model = None
            self.alpha_ = 0.0
            return self
        self._single_class = None
        rng = np.random.default_rng(self.random_state)
        train_idx, val_idx = _train_val_split(y_encoded, float(self.val_fraction), rng)
        if val_idx.size == 0:
            train_idx = np.arange(y_encoded.shape[0], dtype=int)
        tabular_train, seq_train = _split_hybrid_payload(x_checked, tabular_key=self.tabular_key)
        self._tabular_model = ExtraTreesClassifier(
            n_estimators=int(self.n_estimators),
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=int(self.random_state),
        )
        self._deep_model = AssistHybridFusionClassifier(
            tabular_key=self.tabular_key,
            token_dim=int(self.token_dim),
            num_heads=int(self.num_heads),
            num_layers=int(self.num_layers),
            branch_hidden_dim=int(self.branch_hidden_dim),
            tabular_hidden_dim=int(self.tabular_hidden_dim),
            modality_dropout=float(self.modality_dropout),
            batch_size=int(self.batch_size),
            max_epochs=int(self.max_epochs),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
            dropout=float(self.dropout),
            val_fraction=max(0.05, min(0.2, float(self.val_fraction))),
            patience=int(self.patience),
            device=str(self.device),
            random_state=int(self.random_state),
            use_amp=bool(self.use_amp),
        )
        if val_idx.size:
            self._tabular_model.fit(tabular_train[train_idx], y[train_idx])
            self._deep_model.fit(_slice_input_rows_with_payload(seq_train, train_idx, tabular_train, self.tabular_key), y[train_idx])
            tab_val = _align_class_probabilities(
                self._tabular_model,
                tabular_train[val_idx],
                class_labels=self.classes_,
            )
            deep_val = _align_class_probabilities(
                self._deep_model,
                _slice_input_rows_with_payload(seq_train, val_idx, tabular_train, self.tabular_key),
                class_labels=self.classes_,
            )
            self.alpha_ = _select_blend_weight(
                tab_val,
                deep_val,
                y[val_idx],
                alpha_grid=self.alpha_grid,
                class_labels=self.classes_,
            )
        else:
            self.alpha_ = 0.5
        self._tabular_model.fit(tabular_train, y)
        self._deep_model.fit(x_checked, y)
        return self

    def predict_proba(self, x: ArrayInput) -> np.ndarray:
        x_checked = _validate_sequence_input(x)
        if not isinstance(x_checked, dict):
            raise ValueError("assist stacked fusion classifier expects a hybrid input mapping")
        n_rows = input_row_count(x_checked)
        if self._single_class is not None:
            probs = np.zeros((n_rows, 1), dtype=np.float32)
            probs[:, 0] = 1.0
            return probs
        assert self._tabular_model is not None
        assert self._deep_model is not None
        tabular, _ = _split_hybrid_payload(x_checked, tabular_key=self.tabular_key)
        probs_tab = _align_class_probabilities(self._tabular_model, tabular, class_labels=self.classes_)
        probs_deep = _align_class_probabilities(self._deep_model, x_checked, class_labels=self.classes_)
        return _blend_probabilities(probs_tab, probs_deep, float(self.alpha_))


class AssistTemporalForestClassifier(BaseEstimator, ClassifierMixin):
    REQUIRED_SEQUENCE_KEYS = AssistHybridFusionClassifier.REQUIRED_SEQUENCE_KEYS

    def __init__(
        self,
        tabular_key: str = "tabular",
        n_estimators: int = 512,
        n_chunks: int = 4,
        random_state: int = 20260415,
    ) -> None:
        self.tabular_key = tabular_key
        self.n_estimators = n_estimators
        self.n_chunks = n_chunks
        self.random_state = random_state

    def prepare_benchmark_input(
        self,
        *,
        tabular: np.ndarray,
        sequence: dict[str, np.ndarray] | None,
        **_: object,
    ) -> dict[str, np.ndarray]:
        if sequence is None:
            raise ValueError("assist temporal forest requires aligned assist sequence payloads")
        payload = {name: np.asarray(value) for name, value in sequence.items()}
        missing = [key for key in self.REQUIRED_SEQUENCE_KEYS if key not in payload]
        if missing:
            raise ValueError(f"assist temporal forest payload is missing required keys: {missing}")
        payload[self.tabular_key] = np.asarray(tabular, dtype=np.float32)
        return payload

    def fit(self, x: ArrayInput, y: np.ndarray) -> "AssistTemporalForestClassifier":
        x_checked = _validate_sequence_input(x)
        if not isinstance(x_checked, dict):
            raise ValueError("assist temporal forest expects a hybrid input mapping")
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        if self.classes_.shape[0] == 1:
            self._single_class = int(self.classes_[0])
            self._estimator = None
            return self
        self._single_class = None
        features = _build_assist_temporal_forest_features(
            x_checked,
            tabular_key=self.tabular_key,
            n_chunks=int(self.n_chunks),
        )
        self._estimator = ExtraTreesClassifier(
            n_estimators=int(self.n_estimators),
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=int(self.random_state),
        )
        self._estimator.fit(features, y)
        self.classes_ = np.asarray(self._estimator.classes_, dtype=int)
        return self

    def predict_proba(self, x: ArrayInput) -> np.ndarray:
        x_checked = _validate_sequence_input(x)
        if not isinstance(x_checked, dict):
            raise ValueError("assist temporal forest expects a hybrid input mapping")
        n_rows = input_row_count(x_checked)
        if self._single_class is not None:
            probs = np.zeros((n_rows, 1), dtype=np.float32)
            probs[:, 0] = 1.0
            return probs
        assert self._estimator is not None
        features = _build_assist_temporal_forest_features(
            x_checked,
            tabular_key=self.tabular_key,
            n_chunks=int(self.n_chunks),
        )
        return np.asarray(self._estimator.predict_proba(features), dtype=np.float32)


class HybridTemporalTabularClassifier(_TorchSequenceClassifier):
    def __init__(
        self,
        payload_key: str,
        mask_key: str | None = None,
        tabular_key: str = "tabular",
        sequence_hidden_dim: int = 160,
        tabular_hidden_dim: int = 256,
        fusion_hidden_dim: int = 256,
        kernel_size: int = 5,
        batch_size: int = 256,
        max_epochs: int = 26,
        lr: float = 1.6e-3,
        weight_decay: float = 2e-4,
        dropout: float = 0.15,
        val_fraction: float = 0.1,
        patience: int = 6,
        grad_clip: float = 1.0,
        device: str = "auto",
        random_state: int = 20260415,
        use_amp: bool = True,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            val_fraction=val_fraction,
            patience=patience,
            grad_clip=grad_clip,
            device=device,
            random_state=random_state,
            use_amp=use_amp,
        )
        self.payload_key = payload_key
        self.mask_key = mask_key
        self.tabular_key = tabular_key
        self.sequence_hidden_dim = sequence_hidden_dim
        self.tabular_hidden_dim = tabular_hidden_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.kernel_size = kernel_size

    def prepare_benchmark_input(
        self,
        *,
        tabular: np.ndarray,
        sequence: dict[str, np.ndarray] | None,
        **_: object,
    ) -> dict[str, np.ndarray]:
        if sequence is None:
            raise ValueError("hybrid temporal/tabular classifier requires aligned sequence payloads")
        payload = {name: np.asarray(value) for name, value in sequence.items()}
        payload[self.tabular_key] = np.asarray(tabular, dtype=np.float32)
        return payload

    def _build_network(self, x_train: ArrayInput, n_classes: int):
        assert isinstance(x_train, dict)
        sequence_dim = int(np.asarray(x_train[self.payload_key]).shape[-1])
        tabular = np.asarray(x_train[self.tabular_key])
        tabular_dim = int(tabular.shape[-1]) if tabular.ndim >= 2 else 1
        return _HybridTemporalTabularNetwork(
            payload_key=self.payload_key,
            mask_key=self.mask_key,
            tabular_key=self.tabular_key,
            sequence_input_dim=sequence_dim,
            tabular_input_dim=tabular_dim,
            sequence_hidden_dim=int(self.sequence_hidden_dim),
            tabular_hidden_dim=int(self.tabular_hidden_dim),
            fusion_hidden_dim=int(self.fusion_hidden_dim),
            n_classes=int(n_classes),
            dropout=float(self.dropout),
            kernel_size=int(self.kernel_size),
        )


class _MaskedTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 5, dropout: float = 0.15):
        super().__init__()
        padding = int(kernel_size) // 2
        self.net = nn.Sequential(
            nn.Conv1d(int(input_dim), int(hidden_dim), kernel_size=int(kernel_size), padding=padding),
            nn.GELU(),
            nn.Conv1d(int(hidden_dim), int(hidden_dim), kernel_size=int(kernel_size), padding=padding),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.out_norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected sequence tensor [batch, time, channels], got {tuple(x.shape)}")
        if mask is not None:
            mask = _reduce_mask(mask).to(dtype=x.dtype)
            x = x * mask
        h = self.net(x.transpose(1, 2)).transpose(1, 2)
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (h * mask).sum(dim=1) / denom
        return self.out_norm(pooled)


class _TemporalSequenceNetwork(nn.Module):
    def __init__(
        self,
        payload_key: str,
        mask_key: str | None,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        dropout: float,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.payload_key = payload_key
        self.mask_key = mask_key
        self.encoder = _MaskedTemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch[self.payload_key]
        mask = batch.get(self.mask_key) if self.mask_key else None
        token = self.encoder(x, mask=mask)
        return self.head(token)


class _HybridTemporalTabularNetwork(nn.Module):
    def __init__(
        self,
        payload_key: str,
        mask_key: str | None,
        tabular_key: str,
        sequence_input_dim: int,
        tabular_input_dim: int,
        sequence_hidden_dim: int,
        tabular_hidden_dim: int,
        fusion_hidden_dim: int,
        n_classes: int,
        dropout: float,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.payload_key = payload_key
        self.mask_key = mask_key
        self.tabular_key = tabular_key
        self.sequence_encoder = _MaskedTemporalEncoder(
            input_dim=sequence_input_dim,
            hidden_dim=sequence_hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_input_dim, tabular_hidden_dim),
            nn.LayerNorm(tabular_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tabular_hidden_dim, tabular_hidden_dim),
            nn.LayerNorm(tabular_hidden_dim),
            nn.GELU(),
        )
        self.sequence_projection = nn.Linear(sequence_hidden_dim, fusion_hidden_dim)
        self.tabular_projection = nn.Linear(tabular_hidden_dim, fusion_hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(sequence_hidden_dim + tabular_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_hidden_dim + sequence_hidden_dim + tabular_hidden_dim),
            nn.Linear(fusion_hidden_dim + sequence_hidden_dim + tabular_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, n_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        sequence = batch[self.payload_key]
        mask = batch.get(self.mask_key) if self.mask_key else None
        tabular = batch[self.tabular_key]
        if tabular.ndim == 3:
            tabular = tabular[:, 0, :]
        elif tabular.ndim == 1:
            tabular = tabular[:, None]
        sequence_token = self.sequence_encoder(sequence, mask=mask)
        tabular_token = self.tabular_branch(tabular)
        sequence_proj = self.sequence_projection(sequence_token)
        tabular_proj = self.tabular_projection(tabular_token)
        gate = self.gate(torch.cat([sequence_token, tabular_token], dim=1))
        fused = gate * sequence_proj + (1.0 - gate) * tabular_proj
        return self.head(torch.cat([sequence_token, tabular_token, fused], dim=1))


class _PACTFormerNetwork(nn.Module):
    MODALITIES = (
        ("assist_emg", "assist_time_mask"),
        ("assist_gaze", "assist_gaze_mask"),
        ("assist_pupil", "assist_pupil_mask"),
        ("assist_acc", "assist_acc_mask"),
        ("assist_gyr", "assist_gyr_mask"),
        ("assist_context", "assist_context_mask"),
    )

    def __init__(
        self,
        modality_dims: dict[str, int],
        token_dim: int,
        num_heads: int,
        num_layers: int,
        branch_hidden_dim: int,
        n_classes: int,
        dropout: float,
        modality_dropout: float,
    ) -> None:
        super().__init__()
        self.modality_dropout = float(modality_dropout)
        self.encoders = nn.ModuleDict(
            {
                name: _MaskedTemporalEncoder(
                    input_dim=int(modality_dims[name]),
                    hidden_dim=int(branch_hidden_dim),
                    kernel_size=5,
                    dropout=dropout,
                )
                for name, _ in self.MODALITIES
            }
        )
        self.projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(int(branch_hidden_dim), int(token_dim)),
                    nn.LayerNorm(int(token_dim)),
                    nn.GELU(),
                )
                for name, _ in self.MODALITIES
            }
        )
        self.modality_embeddings = nn.Parameter(torch.randn(len(self.MODALITIES) + 1, int(token_dim)) * 0.02)
        self.available_embedding = nn.Parameter(torch.randn(1, 1, int(token_dim)) * 0.02)
        self.missing_embedding = nn.Parameter(torch.randn(1, 1, int(token_dim)) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, int(token_dim)) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(token_dim),
            nhead=int(num_heads),
            dim_feedforward=int(token_dim) * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.head = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(token_dim), int(n_classes)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens: list[torch.Tensor] = []
        availability_terms: list[torch.Tensor] = []
        for offset, (name, mask_name) in enumerate(self.MODALITIES, start=1):
            x = batch[name]
            mask = batch.get(mask_name)
            if mask is not None:
                reduced_mask = _reduce_mask(mask)
                available = reduced_mask.amax(dim=(1, 2))
            else:
                available = torch.ones(x.shape[0], device=x.device)
            encoded = self.encoders[name](x, mask=mask)
            token = self.projectors[name](encoded)
            token = token + self.modality_embeddings[offset]
            tokens.append(token)
            availability_terms.append(available[:, None, None])
        token_tensor = torch.stack(tokens, dim=1)
        availability = torch.cat(availability_terms, dim=1).clamp(0.0, 1.0)
        if self.training and self.modality_dropout > 0.0:
            dropout_mask = (torch.rand_like(availability) > self.modality_dropout).to(token_tensor.dtype)
            availability = availability * dropout_mask
        token_tensor = token_tensor + availability * self.available_embedding + (1.0 - availability) * self.missing_embedding
        cls = self.cls_token.expand(token_tensor.shape[0], -1, -1) + self.modality_embeddings[:1]
        encoded = self.transformer(torch.cat([cls, token_tensor], dim=1))
        cls_out = encoded[:, 0]
        logits = self.head(cls_out)
        consistency = 1e-3 * token_tensor.square().mean()
        return logits, consistency


class _AssistHybridFusionNetwork(nn.Module):
    MODALITIES = _PACTFormerNetwork.MODALITIES

    def __init__(
        self,
        *,
        modality_dims: dict[str, int],
        tabular_key: str,
        tabular_dim: int,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        branch_hidden_dim: int,
        tabular_hidden_dim: int,
        n_classes: int,
        dropout: float,
        modality_dropout: float,
    ) -> None:
        super().__init__()
        self.tabular_key = tabular_key
        self.modality_dropout = float(modality_dropout)
        self.encoders = nn.ModuleDict(
            {
                name: _MaskedTemporalEncoder(
                    input_dim=int(modality_dims[name]),
                    hidden_dim=int(branch_hidden_dim),
                    kernel_size=5,
                    dropout=dropout,
                )
                for name, _ in self.MODALITIES
            }
        )
        self.projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(int(branch_hidden_dim), int(token_dim)),
                    nn.LayerNorm(int(token_dim)),
                    nn.GELU(),
                )
                for name, _ in self.MODALITIES
            }
        )
        self.tabular_branch = nn.Sequential(
            nn.Linear(int(tabular_dim), int(tabular_hidden_dim)),
            nn.LayerNorm(int(tabular_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(tabular_hidden_dim), int(token_dim)),
            nn.LayerNorm(int(token_dim)),
            nn.GELU(),
        )
        self.modality_gate = nn.Sequential(
            nn.Linear(int(token_dim), int(token_dim)),
            nn.GELU(),
            nn.Linear(int(token_dim), 1),
        )
        self.modality_embeddings = nn.Parameter(torch.randn(len(self.MODALITIES) + 3, int(token_dim)) * 0.02)
        self.available_embedding = nn.Parameter(torch.randn(1, 1, int(token_dim)) * 0.02)
        self.missing_embedding = nn.Parameter(torch.randn(1, 1, int(token_dim)) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, int(token_dim)) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(token_dim),
            nhead=int(num_heads),
            dim_feedforward=int(token_dim) * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.head = nn.Sequential(
            nn.LayerNorm(int(token_dim) * 3),
            nn.Linear(int(token_dim) * 3, int(token_dim) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(token_dim) * 2, int(n_classes)),
        )
        self.tabular_skip = nn.Linear(int(token_dim), int(n_classes))
        self.sequence_skip = nn.Linear(int(token_dim), int(n_classes))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        tabular = batch[self.tabular_key]
        if tabular.ndim == 3:
            tabular = tabular[:, 0, :]
        elif tabular.ndim == 1:
            tabular = tabular[:, None]
        tabular_token = self.tabular_branch(tabular) + self.modality_embeddings[1]
        tokens: list[torch.Tensor] = []
        availability_terms: list[torch.Tensor] = []
        for offset, (name, mask_name) in enumerate(self.MODALITIES, start=3):
            x = batch[name]
            mask = batch.get(mask_name)
            if mask is not None:
                reduced_mask = _reduce_mask(mask)
                available = reduced_mask.amax(dim=(1, 2))
            else:
                available = torch.ones(x.shape[0], device=x.device)
            encoded = self.encoders[name](x, mask=mask)
            token = self.projectors[name](encoded) + self.modality_embeddings[offset]
            tokens.append(token)
            availability_terms.append(available[:, None, None])
        token_tensor = torch.stack(tokens, dim=1)
        availability = torch.cat(availability_terms, dim=1).clamp(0.0, 1.0)
        if self.training and self.modality_dropout > 0.0:
            dropout_mask = (torch.rand_like(availability) > self.modality_dropout).to(token_tensor.dtype)
            availability = availability * dropout_mask
        token_tensor = token_tensor + availability * self.available_embedding + (1.0 - availability) * self.missing_embedding
        gate_logits = self.modality_gate(token_tensor).squeeze(-1)
        gate_logits = gate_logits.masked_fill(availability.squeeze(-1) <= 0.0, -1e4)
        gate_weights = torch.softmax(gate_logits, dim=1).unsqueeze(-1)
        pooled_token = torch.sum(gate_weights * token_tensor, dim=1) + self.modality_embeddings[2]
        cls = self.cls_token.expand(token_tensor.shape[0], -1, -1) + self.modality_embeddings[:1]
        stacked = torch.cat([cls, tabular_token[:, None, :], pooled_token[:, None, :], token_tensor], dim=1)
        encoded = self.transformer(stacked)
        cls_out = encoded[:, 0]
        tabular_out = encoded[:, 1]
        pooled_out = encoded[:, 2]
        logits = self.head(torch.cat([cls_out, tabular_out, pooled_out], dim=1))
        logits = logits + 0.35 * self.tabular_skip(tabular_token) + 0.20 * self.sequence_skip(pooled_token)
        consistency = 1e-3 * gate_weights.square().mean()
        return logits, consistency


def _validate_sequence_input(x: ArrayInput) -> ArrayInput:
    if isinstance(x, dict):
        if not x:
            raise ValueError("sequence input payload is empty")
        input_row_count(x)
        return {name: np.asarray(value) for name, value in x.items()}
    return np.asarray(x)


def _fit_input_normalizers(x: ArrayInput) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not isinstance(x, dict):
        return {}
    normalizers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, value in x.items():
        arr = np.asarray(value, dtype=np.float32)
        if name.endswith("_mask"):
            continue
        if arr.ndim == 3:
            mean = arr.mean(axis=(0, 1), keepdims=True)
            std = arr.std(axis=(0, 1), keepdims=True)
        elif arr.ndim == 2:
            mean = arr.mean(axis=0, keepdims=True)
            std = arr.std(axis=0, keepdims=True)
        else:
            continue
        std = np.where(std < 1e-6, 1.0, std)
        normalizers[name] = (mean.astype(np.float32), std.astype(np.float32))
    return normalizers


def _split_hybrid_payload(payload: dict[str, np.ndarray], *, tabular_key: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if tabular_key not in payload:
        raise ValueError(f"hybrid payload is missing tabular key '{tabular_key}'")
    tabular = np.asarray(payload[tabular_key], dtype=np.float32)
    sequence = {
        name: np.asarray(value)
        for name, value in payload.items()
        if name != tabular_key
    }
    if not sequence:
        raise ValueError("hybrid payload must include at least one non-tabular sequence tensor")
    return tabular, sequence


def _slice_input_rows_with_payload(
    sequence_payload: dict[str, np.ndarray],
    index: np.ndarray,
    tabular_payload: np.ndarray,
    tabular_key: str,
) -> dict[str, np.ndarray]:
    sliced = {name: np.asarray(value)[index] for name, value in sequence_payload.items()}
    sliced[tabular_key] = np.asarray(tabular_payload)[index]
    return sliced


def _align_class_probabilities(model: object, x: ArrayInput, *, class_labels: np.ndarray) -> np.ndarray:
    probs = np.asarray(model.predict_proba(x), dtype=np.float32)
    classes_attr = getattr(model, "classes_", None)
    if classes_attr is None:
        return probs
    classes = np.asarray(classes_attr, dtype=int)
    full = np.zeros((input_row_count(x), int(class_labels.shape[0])), dtype=np.float32)
    class_to_col = {int(label): idx for idx, label in enumerate(np.asarray(class_labels, dtype=int))}
    for src_col, label in enumerate(classes):
        full[:, class_to_col[int(label)]] = probs[:, src_col]
    row_sum = full.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return full / row_sum


def _blend_probabilities(probs_a: np.ndarray, probs_b: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = (1.0 - alpha) * np.asarray(probs_a, dtype=np.float32) + alpha * np.asarray(probs_b, dtype=np.float32)
    row_sum = blended.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return blended / row_sum


def _select_blend_weight(
    probs_tabular: np.ndarray,
    probs_deep: np.ndarray,
    y_true: np.ndarray,
    *,
    alpha_grid: tuple[float, ...],
    class_labels: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    best_alpha = 0.0
    best_key: tuple[float, float, float] | None = None
    labels = np.asarray(class_labels, dtype=int)
    for alpha in alpha_grid:
        blended = _blend_probabilities(probs_tabular, probs_deep, float(alpha))
        pred = blended.argmax(axis=1)
        macro = float(f1_score(y_true, pred, average="macro"))
        nll = float(log_loss(y_true, blended, labels=labels))
        agreement = float(np.mean(pred == y_true))
        rank_key = (macro, agreement, -nll)
        if best_key is None or rank_key > best_key:
            best_key = rank_key
            best_alpha = float(alpha)
    return best_alpha


def _build_assist_temporal_forest_features(
    payload: dict[str, np.ndarray],
    *,
    tabular_key: str,
    n_chunks: int,
) -> np.ndarray:
    tabular, sequence = _split_hybrid_payload(payload, tabular_key=tabular_key)
    feature_blocks = [np.asarray(tabular, dtype=np.float32)]
    feature_blocks.append(
        _sequence_summary_block(
            np.abs(np.asarray(sequence["assist_emg"], dtype=np.float32)),
            np.asarray(sequence["assist_time_mask"], dtype=np.float32),
            n_chunks=n_chunks,
        )
    )
    for name, mask_name in (
        ("assist_gaze", "assist_gaze_mask"),
        ("assist_pupil", "assist_pupil_mask"),
        ("assist_acc", "assist_acc_mask"),
        ("assist_gyr", "assist_gyr_mask"),
        ("assist_context", "assist_context_mask"),
    ):
        feature_blocks.append(
            _sequence_summary_block(
                np.asarray(sequence[name], dtype=np.float32),
                np.asarray(sequence[mask_name], dtype=np.float32),
                n_chunks=n_chunks,
            )
        )
    return np.concatenate(feature_blocks, axis=1)


def _sequence_summary_block(arr: np.ndarray, mask: np.ndarray, *, n_chunks: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        arr = _ensure_tensor_rank(arr)
    reduced_mask = _reduce_numpy_mask(mask)
    counts = reduced_mask.sum(axis=1).clip(min=1.0)
    mean = (arr * reduced_mask).sum(axis=1) / counts
    centered = (arr - mean[:, None, :]) * reduced_mask
    std = np.sqrt((centered * centered).sum(axis=1) / counts)
    first = _masked_endpoint(arr, reduced_mask, from_end=False)
    last = _masked_endpoint(arr, reduced_mask, from_end=True)
    delta = last - first
    availability = counts[:, 0] / max(1, arr.shape[1])
    blocks = [mean, std, delta, availability[:, None].astype(np.float32)]
    chunk_edges = np.linspace(0, arr.shape[1], num=max(2, int(n_chunks) + 1), dtype=int)
    for start, stop in zip(chunk_edges[:-1], chunk_edges[1:], strict=True):
        if stop <= start:
            continue
        chunk = arr[:, start:stop, :]
        chunk_mask = reduced_mask[:, start:stop, :]
        chunk_counts = chunk_mask.sum(axis=1).clip(min=1.0)
        chunk_mean = (chunk * chunk_mask).sum(axis=1) / chunk_counts
        chunk_availability = chunk_counts[:, 0] / max(1, stop - start)
        blocks.append(chunk_mean)
        blocks.append(chunk_availability[:, None].astype(np.float32))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def _reduce_numpy_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"expected mask with 2 or 3 dims, got {arr.shape}")
    if arr.shape[-1] > 1:
        arr = arr.mean(axis=-1, keepdims=True)
    return np.clip(arr, 0.0, 1.0)


def _masked_endpoint(arr: np.ndarray, mask: np.ndarray, *, from_end: bool) -> np.ndarray:
    mask_bool = mask[..., 0] > 0
    if from_end:
        valid_idx = np.where(mask_bool, np.arange(arr.shape[1], dtype=int)[None, :], -1).max(axis=1)
    else:
        valid_idx = np.where(mask_bool, np.arange(arr.shape[1], dtype=int)[None, :], arr.shape[1]).min(axis=1)
        valid_idx = np.where(valid_idx >= arr.shape[1], 0, valid_idx)
    valid_idx = np.clip(valid_idx, 0, max(0, arr.shape[1] - 1))
    rows = np.arange(arr.shape[0], dtype=int)
    return arr[rows, valid_idx, :]


def _apply_normalizers(x: ArrayInput, normalizers: dict[str, tuple[np.ndarray, np.ndarray]]) -> ArrayInput:
    if not isinstance(x, dict):
        return np.asarray(x, dtype=np.float32)
    normalized: dict[str, np.ndarray] = {}
    for name, value in x.items():
        arr = np.asarray(value, dtype=np.float32)
        stats = normalizers.get(name)
        if stats is None:
            normalized[name] = arr
        else:
            mean, std = stats
            normalized[name] = (arr - mean) / std
    return normalized


def _prepare_torch_batch(
    x: ArrayInput,
    *,
    device,
    normalizers: dict[str, tuple[np.ndarray, np.ndarray]],
) -> torch.Tensor | dict[str, torch.Tensor]:
    normalized = _apply_normalizers(x, normalizers)
    if isinstance(normalized, dict):
        return {
            name: torch.from_numpy(_ensure_tensor_rank(arr)).to(device, non_blocking=True)
            for name, arr in normalized.items()
        }
    return torch.from_numpy(np.asarray(normalized, dtype=np.float32)).to(device, non_blocking=True)


def _ensure_tensor_rank(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        return arr[:, None, :]
    return arr


def _reduce_mask(mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        raise ValueError("mask is required for reduction")
    if mask.ndim == 2:
        return mask[:, :, None]
    if mask.ndim == 3 and mask.shape[-1] > 1:
        return mask.mean(dim=-1, keepdim=True)
    return mask


def _iter_numpy_batches(x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    permutation = rng.permutation(x.shape[0])
    for start in range(0, permutation.shape[0], batch_size):
        batch_index = permutation[start : start + batch_size]
        yield x[batch_index], y[batch_index]


def _iter_index_batches(n_samples: int, batch_size: int, rng: np.random.Generator):
    permutation = rng.permutation(n_samples)
    for start in range(0, permutation.shape[0], batch_size):
        yield permutation[start : start + batch_size]


def _iter_ordered_index_batches(n_samples: int, batch_size: int):
    order = np.arange(n_samples, dtype=int)
    for start in range(0, n_samples, batch_size):
        yield order[start : start + batch_size]


def _train_val_split(y: np.ndarray, val_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(y.shape[0])
    if val_fraction <= 0 or n_samples < 256:
        index = np.arange(n_samples, dtype=int)
        return index, np.asarray([], dtype=int)
    min_class = int(np.min(np.bincount(y)))
    stratify = y if min_class >= 2 else None
    all_index = np.arange(n_samples, dtype=int)
    train_idx, val_idx = train_test_split(
        all_index,
        test_size=float(val_fraction),
        random_state=int(rng.integers(0, 2**31 - 1)),
        stratify=stratify,
    )
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _balanced_class_weights(y: np.ndarray, n_classes: int):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (float(n_classes) * counts)
    return torch.from_numpy(weights.astype(np.float32))


def _validation_loss_numpy(network, criterion, x_val, y_val, batch_size: int, device, use_amp: bool) -> float:
    if x_val is None or y_val is None or x_val.shape[0] == 0:
        return float("inf")
    network.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, x_val.shape[0], batch_size):
            stop = min(start + batch_size, x_val.shape[0])
            batch_x = torch.from_numpy(x_val[start:stop]).to(device, non_blocking=True)
            batch_y = torch.from_numpy(y_val[start:stop]).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = network(batch_x)
                loss = criterion(logits, batch_y)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _validation_loss_sequence(
    network,
    criterion,
    x_val: ArrayInput | None,
    y_val: np.ndarray | None,
    *,
    batch_size: int,
    normalizers: dict[str, tuple[np.ndarray, np.ndarray]],
    device,
    use_amp: bool,
) -> float:
    if x_val is None or y_val is None or y_val.shape[0] == 0:
        return float("inf")
    network.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for batch_index in _iter_ordered_index_batches(y_val.shape[0], batch_size):
            batch_x = _prepare_torch_batch(
                slice_input_rows(x_val, batch_index),
                device=device,
                normalizers=normalizers,
            )
            batch_y = torch.from_numpy(y_val[batch_index]).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, aux_loss = _unpack_network_output(network(batch_x))
                loss = criterion(logits, batch_y) + aux_loss
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _unpack_network_output(output) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, tuple):
        logits, aux_loss = output
        if not torch.is_tensor(aux_loss):
            aux_loss = torch.tensor(float(aux_loss), device=logits.device)
        aux_loss = torch.nan_to_num(aux_loss, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0), aux_loss
    return torch.nan_to_num(output, nan=0.0, posinf=20.0, neginf=-20.0), torch.tensor(0.0, device=output.device)


def _configure_torch(device, random_state: int) -> None:
    if torch is None:
        return
    torch.manual_seed(int(random_state))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(random_state))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def _build_grad_scaler(enabled: bool):
    if torch is None:
        raise ImportError("torch is required for torch models")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)

