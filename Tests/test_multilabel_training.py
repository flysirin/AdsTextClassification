import json
import math
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
import torch
from sklearn.feature_extraction.text import CountVectorizer

from AppFlaskPyTorch.app import create_app
from config import config
from models.model import SimpleNN
from utils.dataset_split import file_sha256, prepare_dataset
from utils.training import calculate_pos_weight, encode_targets, save_model, train_model


ROOT = Path(__file__).resolve().parents[1]


# Управляемая валидация проверяет остановку и восстановление независимо от сходимости.
@pytest.mark.parametrize("sequence,patience,delta,enabled,count,best,reason", [
    ([1.0, 0.8, 0.9, 1.0], 2, 0.0, True, 4, 2, "patience"),
    ([1.0, 1.0, 1.0, 1.0], 2, 0.0, True, 3, 1, "patience"),
    ([1.0, 0.98, 0.97, 0.96], 2, 0.1, True, 3, 3, "patience"),
    ([1.0, 0.94, 0.88, 0.87], 2, 0.1, True, 4, 4, "max_epochs"),
    ([1.0, 0.8, 0.9, 1.0], 1, 0.0, False, 4, 2, "max_epochs"),
    ([1.0, 0.9, 0.8, 0.7], 2, 0.0, True, 4, 4, "max_epochs"),
])
def test_best_epoch_and_stopping(tmp_path, monkeypatch, sequence, patience, delta, enabled, count, best, reason):
    import utils.training as training
    source = tmp_path / "data.csv"
    samples().to_csv(source, index=False)
    split = prepare_dataset(source)
    states = []

    def controlled_validation(model, vectors, targets, batch_size, device):
        assert model.training
        states.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
        return sequence[len(states) - 1]

    monkeypatch.setattr(training, "validation_bce", controlled_validation)
    result = train_model(split, multilabel=True, epochs=len(sequence), training_mode="best",
                         patience=patience, min_delta=delta, early_stopping=enabled)
    history = result.vectorizer.training_history_
    assert history["epochs_completed"] == count
    assert history["best_epoch"] == best
    assert history["stop_reason"] == reason
    assert history["validation_losses"] == sequence[:count]
    assert result.validation_loss == min(sequence[:count])
    # Независимая копия лучшей эпохи переживает последующие шаги и запись на диск.
    save_model(result, tmp_path / "candidate")
    saved = torch.load(tmp_path / "candidate/model.pth", weights_only=True)
    assert all(torch.equal(value, saved[key]) for key, value in states[best - 1].items())
    # При одинаковом seed воспроизводятся потери и выбранные веса.
    states.clear()
    repeat = train_model(split, multilabel=True, epochs=len(sequence), training_mode="best",
                         patience=patience, min_delta=delta, early_stopping=enabled)
    assert repeat.losses == result.losses
    assert repeat.vectorizer.training_history_ == history


# Реальная валидация не меняет веса, возвращает режим и отклоняет нечисловые потери.
def test_validation_bce_and_invalid_stopping(tmp_path):
    from utils.training import validation_bce
    source = tmp_path / "data.csv"
    samples().to_csv(source, index=False)
    split = prepare_dataset(source)
    result = train_model(split, multilabel=True, epochs=2, training_mode="best", pos_weight=True)
    _, targets = encode_targets(split.partitions)
    vectors = result.vectorizer.transform(split.partitions["validation"].text)
    result.model.train()
    state = {key: value.clone() for key, value in result.model.state_dict().items()}
    loss = validation_bce(result.model, vectors, targets["validation"], 7, "cpu")
    assert loss == pytest.approx(result.validation_loss, abs=1e-6)
    assert result.model.training
    assert all(torch.equal(value, result.model.state_dict()[key]) for key, value in state.items())
    # Неверные параметры отвергаются до начала обучения.
    for options in ({"patience": 0}, {"min_delta": -1}, {"min_delta": float("nan")}):
        with pytest.raises(ValueError, match="patience"):
            train_model(split, multilabel=True, **options)
    # Повреждённые логиты не должны приводить к сохранению кандидата.
    with torch.no_grad():
        result.model.fc2.bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="нечисловыми"):
        validation_bce(result.model, vectors, targets["validation"], 7, "cpu")
    assert result.model.training


# Синтетические примеры позволяют отдельно проверить обе категории и их сочетание.
def samples():
    records = []
    for index in range(30):
        records.extend([
            {"text": f"Диван мебель образецмебель{index}", "label": "home"},
            {"text": f"Ноутбук электроника образецтехника{index}", "label": "electronics"},
            {"text": f"Диван мебель ноутбук электроника образецвместе{index}", "label": "home|electronics"},
        ])
    return pd.DataFrame(records)


# Цели имеют устойчивый порядок и float32; категории только из контроля отклоняются.
def test_multi_hot_and_untrained_category_rejected():
    parts = {"train": pd.DataFrame({"labels": [["home"], ["electronics", "home"]]}),
             "validation": pd.DataFrame({"labels": [["electronics"]]}),
             "test": pd.DataFrame({"labels": [["home"]]})}
    names, targets = encode_targets(parts)
    assert names == {0: "electronics", 1: "home"}
    assert targets["train"].dtype == torch.float32
    assert targets["train"].tolist() == [[0, 1], [1, 1]]
    # Необученный выход не должен появиться молча из тестовой разметки.
    parts["test"] = pd.DataFrame({"labels": [["jobs"]]})
    with pytest.raises(ValueError, match="без положительных примеров в train: jobs"):
        encode_targets(parts)


# Веса учитывают дисбаланс и не обнуляют обучение категории во всех строках.
def test_pos_weight_and_universal_category_keep_learning_signal():
    targets = torch.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=torch.float32)
    assert calculate_pos_weight(targets).tolist() == [3, 1, 1]
    # Для категории без положительных примеров невозможно обучить нужный выход.
    with pytest.raises(ValueError, match="без положительных примеров"):
        calculate_pos_weight(torch.zeros((4, 1)))


# Повторяемое CPU-обучение должно выучить одиночные и совместные метки.
def test_learns_joint_labels_and_reproduces_saved_model(tmp_path):
    source = tmp_path / "data.csv"
    samples().to_csv(source, index=False)
    split = prepare_dataset(source)
    result = train_model(split, multilabel=True, epochs=160, batch_size=24, source_sha256=file_sha256(source))
    repeat = train_model(split, multilabel=True, epochs=160, batch_size=24, source_sha256=file_sha256(source))
    assert all(math.isfinite(loss) for loss in result.losses)
    assert result.losses[-1] < result.losses[0] / 2
    assert math.isfinite(result.validation_loss)
    assert result.losses == repeat.losses
    assert all(torch.equal(value, repeat.model.state_dict()[key]) for key, value in result.model.state_dict().items())
    # Независимые выходы распознают оба предмета в одном объявлении.
    queries = ["Диван мебель", "Ноутбук электроника", "Диван мебель ноутбук электроника"]
    vectors = torch.tensor(result.vectorizer.transform(queries).toarray(), dtype=torch.float32)
    with torch.inference_mode():
        predictions = result.model(vectors).sigmoid().ge(0.5).int().tolist()
    assert predictions == [[0, 1], [1, 0], [1, 1]]

    # Комплект сохраняет три файла и параметры, достаточные для воспроизведения.
    directory = tmp_path / "candidate"
    save_model(result, directory)
    assert {file.name for file in directory.iterdir()} == {"model.pth", "vectorizer.pkl", "labels_encoded.json"}
    labels = json.loads((directory / "labels_encoded.json").read_text(encoding="utf-8"))
    assert labels == {"0": "electronics", "1": "home"}
    saved_vectorizer = joblib.load(directory / "vectorizer.pkl")
    assert saved_vectorizer.classification_task_ == "multilabel"
    metadata = saved_vectorizer.training_config_
    assert metadata["source_sha256"] == file_sha256(source)
    assert metadata["partition_records"] == split.manifest["partitions"]
    assert metadata["seed"] == 42
    assert metadata["loss"] == "BCEWithLogitsLoss"
    assert metadata["batch_size"] == 24
    assert metadata["pos_weight"] is None
    assert saved_vectorizer.training_history_["losses"] == result.losses
    # Прежняя структура весов загружается с теми же логитами на CPU.
    state = torch.load(directory / "model.pth", map_location="cpu", weights_only=True)
    assert set(state) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    restored = SimpleNN(len(saved_vectorizer.vocabulary_), 100, len(labels))
    restored.load_state_dict(state)
    restored.eval()
    with torch.inference_mode():
        assert torch.equal(restored(vectors), result.model(vectors))
    # Повторное сохранение не должно перезаписать уже готовый кандидат.
    with pytest.raises(FileExistsError):
        save_model(result, directory)


# CLI обучает все поддерживаемые форматы и помечает кандидат для защиты загрузчиков.
@pytest.mark.parametrize("ext,sep,weighted", [("csv", ",", False), ("csv", ";", True), ("xlsx", None, False)])
def test_multilabel_command_and_legacy_readers_fail_closed(tmp_path, ext, sep, weighted):
    source = tmp_path / f"data.{ext}"
    if sep:
        samples().to_csv(source, sep=sep, index=False)
    else:
        samples().to_excel(source, index=False)
    original = source.read_bytes()
    directory = tmp_path / "candidate"
    command = [sys.executable, "train.py", "--input", source, "--model-dir", directory,
               "--multilabel", "--epochs", "1", "--batch-size", "7"]
    if weighted:
        command.append("--pos-weight")
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    assert "Validation BCE=" in completed.stdout
    assert "Accuracy=" not in completed.stdout
    vectorizer = joblib.load(directory / "vectorizer.pkl")
    assert vectorizer.classification_task_ == "multilabel"
    assert math.isfinite(vectorizer.training_history_["losses"][0])
    assert math.isfinite(vectorizer.training_history_["validation_bce"])
    # Веса и словарь зависят только от обучающей части.
    split = prepare_dataset(source)
    _, targets = encode_targets(split.partitions)
    expected = calculate_pos_weight(targets["train"]).tolist() if weighted else None
    assert vectorizer.training_config_["pos_weight"] == expected
    analyzer = vectorizer.build_analyzer()
    train_tokens = {token for text in split.partitions["train"].text for token in analyzer(text)}
    assert set(vectorizer.vocabulary_) == train_tokens

    # Прежний API отклоняет multilabel вместо некорректного выбора через argmax.
    client = create_app(directory).test_client()
    health = client.get("/health").json
    assert not health["model_ready"]
    assert health["error"] == "model_load_failed"
    assert client.post("/predict", json={"messages": {"id": "Диван"}}).status_code == 503
    # Пакетный прогноз тоже отказывает до записи и сохраняет существующий результат.
    output = tmp_path / f"protected.{ext}"
    output.write_bytes(b"do not overwrite")
    completed = subprocess.run([sys.executable, "run_predict_data.py", "--input", source,
                                "--model-dir", directory, "--output", output],
                               cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 2
    assert "argmax" in completed.stderr
    assert output.read_bytes() == b"do not overwrite"
    assert source.read_bytes() == original


# Старый комплект без метаданных режима остаётся совместимым с API и ID сообщений.
def test_legacy_vectorizer_without_metadata_still_loads(tmp_path):
    vectorizer = CountVectorizer().fit(["Диван мебель", "Ноутбук электроника"])
    joblib.dump(vectorizer, tmp_path / "vectorizer.pkl")
    torch.save(SimpleNN(len(vectorizer.vocabulary_), 100, 2).state_dict(), tmp_path / "model.pth")
    (tmp_path / "labels_encoded.json").write_text(json.dumps({0: "electronics", 1: "home"}), encoding="utf-8")
    client = create_app(tmp_path).test_client()
    assert client.get("/health").json["model_ready"]
    response = client.post("/predict", json={"messages": {"id-2": "Ноутбук", "id-1": "Диван"}})
    assert response.status_code == 200
    assert set(response.json["predictions"]) == {"id-1", "id-2"}


# Новое обучение не должно затронуть рабочую модель или существующий каталог.
@pytest.mark.parametrize("target", ["missing", "deployed", "nested", "existing"])
def test_multilabel_candidate_directory_is_protected(tmp_path, monkeypatch, target):
    from train import main

    deployed = tmp_path / "deployed"
    deployed.mkdir()
    marker = deployed / "model.pth"
    marker.write_bytes(b"existing model")
    monkeypatch.setattr(config, "MODEL_PATH", str(deployed))
    paths = {"deployed": deployed, "nested": deployed / "candidate", "existing": tmp_path}
    arguments = ["train.py", "--multilabel"]
    if target != "missing":
        arguments.extend(["--model-dir", str(paths[target])])
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert marker.read_bytes() == b"existing model"
    assert not (deployed / "candidate").exists()


# Одна GPU-эпоха через штатный Bash-скрипт использует только временные данные.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Для CUDA нужен отдельный профиль и NVIDIA")
@pytest.mark.parametrize("training_mode", ["legacy", "best"])
def test_cuda_smoke_through_training_script(tmp_path, training_mode):
    print(f"CUDA smoke: {torch.cuda.get_device_name(0)}, PyTorch={torch.__version__}, CUDA={torch.version.cuda}")
    source = tmp_path / "data.csv"
    samples().to_csv(source, index=False)
    original = source.read_bytes()
    directory = tmp_path / "candidate"
    subprocess.run(["bash", "scripts/train_cuda.sh", "--input", source, "--model-dir", directory,
                    "--multilabel", "--epochs", "1", "--batch-size", "8",
                    "--training-mode", training_mode], cwd=ROOT, check=True)
    # Проверяем фактическое устройство, конечные потери и переносимость весов на CPU.
    vectorizer = joblib.load(directory / "vectorizer.pkl")
    assert vectorizer.training_config_["device"] == "cuda"
    assert vectorizer.training_config_["training_mode"] == training_mode
    if training_mode == "best":
        assert vectorizer.training_history_["best_epoch"] == 1
    assert math.isfinite(vectorizer.training_history_["losses"][0])
    assert math.isfinite(vectorizer.training_history_["validation_bce"])
    state = torch.load(directory / "model.pth", map_location="cpu", weights_only=True)
    assert all(value.device.type == "cpu" for value in state.values())
    assert source.read_bytes() == original
