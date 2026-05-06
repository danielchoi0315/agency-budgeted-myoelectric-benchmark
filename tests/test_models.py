from __future__ import annotations

import numpy as np
from sklearn.base import clone

from myoagency.models import fit_predict_proba, shallow_model_specs


def _make_sequence_split(
    rng: np.random.Generator,
    labels: np.ndarray,
    *,
    max_length: int = 24,
    n_channels: int = 8,
    context_dim: int = 5,
) -> dict[str, np.ndarray]:
    n_samples = int(labels.shape[0])
    user_emg = np.zeros((n_samples, max_length, n_channels), dtype=np.float32)
    assist_emg = np.zeros((n_samples, max_length, n_channels), dtype=np.float32)
    assist_gaze = np.zeros((n_samples, max_length, 2), dtype=np.float32)
    assist_pupil = np.zeros((n_samples, max_length, 2), dtype=np.float32)
    assist_acc = np.zeros((n_samples, max_length, 3), dtype=np.float32)
    assist_gyr = np.zeros((n_samples, max_length, 3), dtype=np.float32)
    assist_context = np.zeros((n_samples, max_length, context_dim), dtype=np.float32)
    time_mask = np.zeros((n_samples, max_length, 1), dtype=np.float32)
    assist_pupil_mask = np.zeros((n_samples, max_length, 2), dtype=np.float32)
    lengths = rng.integers(max_length // 2, max_length + 1, size=n_samples)
    full_time = np.linspace(0.0, 2.0 * np.pi, max_length, dtype=np.float32)
    for row_idx, (label, length) in enumerate(zip(labels, lengths, strict=True)):
        time_mask[row_idx, :length] = 1.0
        assist_pupil_mask[row_idx, :length] = 1.0
        waveform = np.stack(
            [
                np.sin(full_time[:length] + 0.6 * int(label)),
                np.cos(full_time[:length] * (1.0 + 0.2 * int(label))),
            ],
            axis=1,
        )
        noise_block = rng.normal(
            scale=0.10,
            size=(int(length), n_channels),
        ).astype(np.float32)
        user_emg[row_idx, :length] = noise_block
        assist_emg[row_idx, :length] = noise_block
        user_emg[row_idx, :length, int(label)] += 1.75
        assist_emg[row_idx, :length, int(label)] += 2.0
        assist_emg[row_idx, :length, int(label) + 3] += 0.85
        user_emg[row_idx, :length, :2] += waveform
        assist_emg[row_idx, :length, :2] += waveform
        assist_gaze[row_idx, :length, 0] = np.linspace(0.1, 0.9, int(length), dtype=np.float32) + 0.1 * int(label)
        assist_gaze[row_idx, :length, 1] = np.linspace(0.9, 0.1, int(length), dtype=np.float32)
        assist_pupil[row_idx, :length, 0] = 0.5 + 0.25 * int(label)
        assist_pupil[row_idx, :length, 1] = 0.4 + 0.1 * np.sin(full_time[:length] + int(label))
        assist_acc[row_idx, :length, 0] = float(label) + 0.05 * rng.normal(size=int(length))
        assist_acc[row_idx, :length, 1] = np.linspace(-1.0, 1.0, int(length), dtype=np.float32)
        assist_acc[row_idx, :length, 2] = float(length) / float(max_length)
        assist_gyr[row_idx, :length, 0] = np.cos(full_time[:length] + int(label))
        assist_gyr[row_idx, :length, 1] = np.sin(full_time[:length] * 0.5 + int(label))
        assist_gyr[row_idx, :length, 2] = float(label) * 0.5
        assist_context[row_idx, :length] = rng.normal(scale=0.04, size=(int(length), context_dim))
        assist_context[row_idx, :length, int(label)] += 1.2
        assist_context[row_idx, :length, -1] = float(length) / float(max_length)
    return {
        "user_emg": user_emg,
        "user_time_mask": time_mask,
        "assist_emg": assist_emg,
        "assist_time_mask": time_mask.copy(),
        "assist_gaze": assist_gaze,
        "assist_gaze_mask": time_mask.copy(),
        "assist_pupil": assist_pupil,
        "assist_pupil_mask": assist_pupil_mask,
        "assist_acc": assist_acc,
        "assist_acc_mask": time_mask.copy(),
        "assist_gyr": assist_gyr,
        "assist_gyr_mask": time_mask.copy(),
        "assist_context": assist_context,
        "assist_context_mask": time_mask.copy(),
    }


def _make_synthetic_sequence_dataset() -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(17)
    y_train = np.repeat(np.arange(3, dtype=int), 18)
    rng.shuffle(y_train)
    y_test = np.repeat(np.arange(3, dtype=int), 4)
    x_train = _make_sequence_split(rng, y_train)
    x_test = _make_sequence_split(rng, y_test)
    return x_train, y_train, x_test


def _make_assist_tabular_features(y_train: np.ndarray, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(29)
    n_classes = int(np.max(y_train)) + 1
    tab_train = rng.normal(scale=0.06, size=(y_train.shape[0], 16)).astype(np.float32)
    tab_test = rng.normal(scale=0.06, size=(n_test, 16)).astype(np.float32)
    y_test = np.repeat(np.arange(n_classes, dtype=int), n_test // n_classes)
    if y_test.shape[0] < n_test:
        y_test = np.concatenate([y_test, np.arange(n_test - y_test.shape[0], dtype=int)])
    for label in np.unique(y_train):
        tab_train[y_train == label, int(label)] += 1.4
        tab_train[y_train == label, int(label) + n_classes] += 0.9
    for row_idx, label in enumerate(y_test[:n_test]):
        tab_test[row_idx, int(label)] += 1.4
        tab_test[row_idx, int(label) + n_classes] += 0.9
    return tab_train, tab_test


def test_tabular_models_emit_normalized_probabilities() -> None:
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(96, 12))
    y_train = rng.integers(0, 4, size=96)
    x_test = rng.normal(size=(8, 12))
    specs = {spec.name: spec for spec in shallow_model_specs()}
    for model_name in ["lda", "logistic", "extra_trees", "random_forest", "svm_rbf"]:
        estimator = clone(specs[model_name].estimator)
        estimator.fit(x_train, y_train)
        probs = estimator.predict_proba(x_test)
        assert probs.shape == (8, 4)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_torch_mlp_runs_on_cpu_when_requested() -> None:
    rng = np.random.default_rng(9)
    x_train = rng.normal(size=(128, 10)).astype(np.float32)
    y_train = np.repeat(np.arange(4), 32)
    x_test = rng.normal(size=(6, 10)).astype(np.float32)
    specs = {spec.name: spec for spec in shallow_model_specs()}
    if "torch_mlp" not in specs:
        return
    estimator = clone(specs["torch_mlp"].estimator)
    estimator.set_params(
        clf__device="cpu",
        clf__batch_size=32,
        clf__hidden_dims=(32, 16),
        clf__max_epochs=2,
        clf__patience=2,
        clf__val_fraction=0.2,
        clf__use_amp=False,
    )
    estimator.fit(x_train, y_train)
    probs = estimator.predict_proba(x_test)
    assert probs.shape == (6, 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_sequence_user_model_handles_dict_sequence_inputs() -> None:
    specs = {spec.name: spec for spec in shallow_model_specs()}
    if "temporal_cnn" not in specs:
        return
    x_train, y_train, x_test = _make_synthetic_sequence_dataset()
    estimator = clone(specs["temporal_cnn"].estimator)
    estimator.set_params(
        device="cpu",
        batch_size=12,
        hidden_dim=24,
        kernel_size=3,
        max_epochs=2,
        patience=2,
        val_fraction=0.2,
        use_amp=False,
    )
    probs = fit_predict_proba(
        estimator,
        {"user_emg": x_train["user_emg"], "user_time_mask": x_train["user_time_mask"]},
        y_train,
        {"user_emg": x_test["user_emg"], "user_time_mask": x_test["user_time_mask"]},
        class_labels=[0, 1, 2],
    )
    assert probs.shape == (x_test["user_emg"].shape[0], 3)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_hybrid_temporal_mlp_user_model_smoke() -> None:
    specs = {spec.name: spec for spec in shallow_model_specs()}
    if "hybrid_temporal_mlp" not in specs:
        return
    x_train, y_train, x_test = _make_synthetic_sequence_dataset()
    rng = np.random.default_rng(23)
    tab_train = rng.normal(scale=0.08, size=(y_train.shape[0], 10)).astype(np.float32)
    tab_test = rng.normal(scale=0.08, size=(x_test["user_emg"].shape[0], 10)).astype(np.float32)
    for label in np.unique(y_train):
        tab_train[y_train == label, int(label)] += 1.5
    for label in np.unique(np.repeat(np.arange(3, dtype=int), 4)):
        tab_test[np.repeat(np.arange(3, dtype=int), 4) == label, int(label)] += 1.5
    estimator = clone(specs["hybrid_temporal_mlp"].estimator)
    estimator.set_params(
        device="cpu",
        batch_size=12,
        sequence_hidden_dim=32,
        tabular_hidden_dim=24,
        fusion_hidden_dim=32,
        kernel_size=3,
        max_epochs=2,
        patience=2,
        val_fraction=0.2,
        use_amp=False,
    )
    probs = fit_predict_proba(
        estimator,
        {
            "tabular": tab_train,
            "user_emg": x_train["user_emg"],
            "user_time_mask": x_train["user_time_mask"],
        },
        y_train,
        {
            "tabular": tab_test,
            "user_emg": x_test["user_emg"],
            "user_time_mask": x_test["user_time_mask"],
        },
        class_labels=[0, 1, 2],
    )
    assert probs.shape == (x_test["user_emg"].shape[0], 3)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_pact_former_assist_model_smoke() -> None:
    specs = {spec.name: spec for spec in shallow_model_specs()}
    if "pactformer" not in specs:
        return
    x_train, y_train, x_test = _make_synthetic_sequence_dataset()
    estimator = clone(specs["pactformer"].estimator)
    estimator.set_params(
        device="cpu",
        batch_size=12,
        token_dim=32,
        num_heads=4,
        num_layers=2,
        branch_hidden_dim=24,
        max_epochs=2,
        patience=2,
        val_fraction=0.2,
        use_amp=False,
    )
    estimator.fit(
        {
            key: value
            for key, value in x_train.items()
            if key.startswith("assist_")
        },
        y_train,
    )
    probs = estimator.predict_proba(
        {
            key: value
            for key, value in x_test.items()
            if key.startswith("assist_")
        }
    )
    assert probs.shape == (x_test["assist_emg"].shape[0], 3)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_assist_hybrid_fusion_model_smoke() -> None:
    specs = {spec.name: spec for spec in shallow_model_specs()}
    if "assist_hybrid_fusion" not in specs:
        return
    x_train, y_train, x_test = _make_synthetic_sequence_dataset()
    tab_train, tab_test = _make_assist_tabular_features(y_train, x_test["assist_emg"].shape[0])
    estimator = clone(specs["assist_hybrid_fusion"].estimator)
    estimator.set_params(
        device="cpu",
        batch_size=12,
        token_dim=32,
        num_heads=4,
        num_layers=2,
        branch_hidden_dim=24,
        tabular_hidden_dim=32,
        max_epochs=2,
        patience=2,
        val_fraction=0.2,
        use_amp=False,
    )
    probs = fit_predict_proba(
        estimator,
        {"tabular": tab_train, **{key: value for key, value in x_train.items() if key.startswith("assist_")}},
        y_train,
        {"tabular": tab_test, **{key: value for key, value in x_test.items() if key.startswith("assist_")}},
        class_labels=[0, 1, 2],
    )
    assert probs.shape == (x_test["assist_emg"].shape[0], 3)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_assist_temporal_forest_model_smoke() -> None:
    specs = {spec.name: spec for spec in shallow_model_specs()}
    if "assist_temporal_forest" not in specs:
        return
    x_train, y_train, x_test = _make_synthetic_sequence_dataset()
    tab_train, tab_test = _make_assist_tabular_features(y_train, x_test["assist_emg"].shape[0])
    estimator = clone(specs["assist_temporal_forest"].estimator)
    estimator.set_params(n_estimators=64, n_chunks=3)
    probs = fit_predict_proba(
        estimator,
        {"tabular": tab_train, **{key: value for key, value in x_train.items() if key.startswith("assist_")}},
        y_train,
        {"tabular": tab_test, **{key: value for key, value in x_test.items() if key.startswith("assist_")}},
        class_labels=[0, 1, 2],
    )
    assert probs.shape == (x_test["assist_emg"].shape[0], 3)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_assist_stacked_fusion_model_smoke() -> None:
    specs = {spec.name: spec for spec in shallow_model_specs()}
    if "assist_stacked_fusion" not in specs:
        return
    x_train, y_train, x_test = _make_synthetic_sequence_dataset()
    tab_train, tab_test = _make_assist_tabular_features(y_train, x_test["assist_emg"].shape[0])
    estimator = clone(specs["assist_stacked_fusion"].estimator)
    estimator.set_params(
        device="cpu",
        batch_size=12,
        token_dim=32,
        num_heads=4,
        num_layers=2,
        branch_hidden_dim=24,
        tabular_hidden_dim=32,
        max_epochs=2,
        patience=2,
        val_fraction=0.2,
        n_estimators=64,
        use_amp=False,
    )
    probs = fit_predict_proba(
        estimator,
        {"tabular": tab_train, **{key: value for key, value in x_train.items() if key.startswith("assist_")}},
        y_train,
        {"tabular": tab_test, **{key: value for key, value in x_test.items() if key.startswith("assist_")}},
        class_labels=[0, 1, 2],
    )
    assert probs.shape == (x_test["assist_emg"].shape[0], 3)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

