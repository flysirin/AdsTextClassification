"""Пакетное обучение SimpleNN с отдельным режимом multilabel."""

import json
import os
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from models.model import SimpleNN

DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 5
DEFAULT_MIN_DELTA = 0.0001


def validation_bce(model, vectors, targets, batch_size, device):
    """Посчитать BCE без весов, сохранив режим модели."""
    was_training = model.training
    model.eval()
    total = 0.0
    try:
        with torch.inference_mode():
            for offset in range(0, vectors.shape[0], batch_size):
                batch = torch.from_numpy(vectors[offset:offset + batch_size].toarray().astype(np.float32)).to(device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(batch), targets[offset:offset + batch_size].to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Потери валидации стали нечисловыми; модель не сохранена")
                total += loss.item() * len(batch)
        return total / vectors.shape[0]
    finally:
        model.train(was_training)


@dataclass
class TrainingResult:
    model: SimpleNN
    vectorizer: CountVectorizer
    labels: dict[int, str]
    losses: list[float]
    validation_loss: float | None


def encode_targets(partitions):
    """Обучить порядок категорий только по train и вернуть float multi-hot цели."""
    classes = sorted({label for labels in partitions["train"].labels for label in labels})
    if not classes:
        raise ValueError("В обучающей части нет категорий")
    for name, frame in partitions.items():
        unknown = {label for labels in frame.labels for label in labels} - set(classes)
        if unknown:
            raise ValueError(f"Категории {name} без положительных примеров в train: " + ", ".join(sorted(unknown)))
    encoder = MultiLabelBinarizer(classes=classes).fit(partitions["train"].labels)
    targets = {name: torch.from_numpy(encoder.transform(frame.labels).astype(np.float32))
               for name, frame in partitions.items()}
    if torch.any(targets["train"].sum(dim=0) == 0):
        raise ValueError("Каждый выход должен иметь положительные примеры в train")
    return dict(enumerate(classes)), targets


def calculate_pos_weight(targets):
    """Вес отрицательных к положительным примерам только из обучающей части."""
    positive = targets.sum(dim=0)
    if torch.any(positive == 0):
        raise ValueError("Нельзя рассчитать pos_weight без положительных примеров")
    negative = len(targets) - positive
    # При категории в каждой строке вес 0 обнулил бы весь сигнал обучения.
    return torch.where(negative > 0, negative / positive, torch.ones_like(positive))


def train_model(split, *, multilabel=False, epochs=DEFAULT_EPOCHS, device="cpu", seed=42,
                batch_size=128, pos_weight=False, source_sha256=None, verbose=False,
                training_mode="legacy", patience=DEFAULT_PATIENCE, min_delta=DEFAULT_MIN_DELTA,
                early_stopping=True):
    if training_mode not in {"legacy", "best"}:
        raise ValueError("Способ обучения должен быть legacy или best")
    if training_mode == "best" and not multilabel:
        raise ValueError("Выбор лучшей эпохи пока доступен только для multilabel")
    if patience < 1 or not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("patience должен быть положительным, min_delta конечным и неотрицательным")
    if epochs < 1 or batch_size < 1:
        raise ValueError("Число эпох и размер пакета должны быть больше нуля")
    if not 0 <= seed < 2**32:
        raise ValueError("seed должен быть от 0 до 2**32 - 1")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Устройство должно быть cpu или cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA недоступна для текущего окружения или драйвера")
    if pos_weight and not multilabel:
        raise ValueError("pos_weight доступен только в режиме multilabel")
    if not multilabel and any(len(labels) != 1 for frame in split.partitions.values() for labels in frame.labels):
        raise ValueError("Для нескольких категорий включите --multilabel")

    names, targets = encode_targets(split.partitions)
    train = split.partitions["train"]
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(train.text)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = SimpleNN(vectors.shape[1], 100, len(names)).to(device)
    weights = calculate_pos_weight(targets["train"]) if pos_weight else None
    criterion = (torch.nn.BCEWithLogitsLoss(pos_weight=weights.to(device) if weights is not None else None)
                 if multilabel else torch.nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    y = targets["train"] if multilabel else targets["train"].argmax(dim=1)
    losses = []
    validation = split.partitions["validation"]
    validation_vectors = vectorizer.transform(validation.text) if multilabel else None
    validation_losses = []
    best_loss = float("inf")
    significant_loss = float("inf")
    best_state = None
    best_epoch = None
    stale = 0
    stop_reason = "max_epochs"
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for ids in torch.randperm(vectors.shape[0]).split(batch_size):
            batch = torch.from_numpy(vectors[ids.numpy()].toarray().astype(np.float32)).to(device)
            loss = criterion(model(batch), y[ids].to(device))
            if not torch.isfinite(loss):
                raise ValueError("Функция потерь стала нечисловой; модель не сохранена")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        losses.append(total / len(train))
        if training_mode == "best":
            current = validation_bce(model, validation_vectors, targets["validation"], batch_size, device)
            validation_losses.append(current)
            if current < best_loss:
                best_loss, best_epoch = current, epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if significant_loss - current > min_delta:
                significant_loss, stale = current, 0
            else:
                stale += 1
        if verbose:
            details = (f", validation BCE={current:.6f}, best_epoch={best_epoch}, without_improvement={stale}"
                       if training_mode == "best" else "")
            print(f"Epoch {epoch + 1}/{epochs}, train loss={losses[-1]:.6f}{details}", flush=True)
        if training_mode == "best" and early_stopping and stale >= patience:
            stop_reason = "patience"
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    validation_loss = None
    if multilabel:
        # Контрольный BCE без весов и без подбора порогов. Независимый test не вычисляется.
        validation_loss = (best_loss if training_mode == "best" else
                           validation_bce(model, validation_vectors, targets["validation"], batch_size, device))

    vectorizer.classification_task_ = "multilabel" if multilabel else "singlelabel"
    vectorizer.training_config_ = {
        "task": vectorizer.classification_task_, "seed": seed, "epochs": epochs,
        "batch_size": batch_size, "device": device, "learning_rate": 0.001,
        "optimizer": "Adam", "loss": "BCEWithLogitsLoss" if multilabel else "CrossEntropyLoss",
        "hidden_size": 100, "num_threads": 4, "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch_version": str(torch.__version__), "classes": list(names.values()),
        "pos_weight": weights.tolist() if weights is not None else None,
        "source_sha256": source_sha256, "split_seed": split.manifest["seed"],
        "split_version": split.manifest["version"], "normalization": split.manifest["normalization"],
        "partition_records": split.manifest["partitions"],
        "training_mode": training_mode, "patience": patience, "min_delta": min_delta,
        "early_stopping": early_stopping and training_mode == "best",
    }
    vectorizer.training_history_ = {
        "losses": losses, "validation_bce": validation_loss, "validation_losses": validation_losses,
        "epochs_completed": len(losses), "best_epoch": best_epoch,
        "best_validation_bce": best_loss if best_epoch is not None else None, "stop_reason": stop_reason,
    }
    return TrainingResult(model.cpu(), vectorizer, names, losses, validation_loss)


def save_model(result, directory, *, allow_existing=False):
    """Сохранить прежний комплект из трёх файлов; новые кандидаты не перезаписывать."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=allow_existing)
    torch.save(result.model.state_dict(), directory / "model.pth")
    joblib.dump(result.vectorizer, directory / "vectorizer.pkl")
    (directory / "labels_encoded.json").write_text(json.dumps(result.labels, ensure_ascii=False), encoding="utf-8")
