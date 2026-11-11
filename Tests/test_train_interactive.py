import json
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import pytest

from config import config
from scripts.train_interactive import collect_options, default_model_dir, main, training_command


NOW = datetime(2026, 11, 11, 15, 30, 0, tzinfo=ZoneInfo("Asia/Tbilisi"))


# Последовательность ответов заменяет клавиатурный ввод в диалоге.
def answers(*values):
    values = iter(values)
    return lambda prompt: next(values)


# Нестандартные каталоги выявляют случайно зашитые пути и защищают рабочую модель.
def configured_paths(tmp_path, monkeypatch):
    data = tmp_path / "my training tables"
    data.mkdir()
    model = tmp_path / "my saved models" / "working"
    model.mkdir(parents=True)
    (model / "model.pth").write_bytes(b"working model")
    monkeypatch.setattr(config, "TRAIN_DATA_PATH", "my training tables")
    monkeypatch.setattr(config, "MODEL_PATH", "my saved models/working")
    return data, model


# Enter выбирает настройки по умолчанию и каталоги из действующего конфига.
def test_defaults_use_configured_directories_and_labeled_table(tmp_path, monkeypatch):
    data, model = configured_paths(tmp_path, monkeypatch)
    for name in ("another.csv", "labeled.csv", "table.XLSX", "~$open.xlsx", "notes.txt"):
        (data / name).write_text("placeholder", encoding="utf-8")
    options = collect_options(root=tmp_path, input_fn=answers("", "", "", "", ""), now=NOW)
    assert options.multilabel
    assert options.training_mode == "best"
    assert options.input_path == data / "labeled.csv"
    assert options.epochs == 100
    assert options.model_dir == model.with_name("working-multilabel-20261111-153000")
    assert not options.model_dir.exists()
    assert (model / "model.pth").read_bytes() == b"working model"


# Ошибки ввода исправляются повторным вопросом без перезаписи существующих путей.
def test_reprompts_invalid_values_and_protects_existing_paths(tmp_path, monkeypatch, capsys):
    data, model = configured_paths(tmp_path, monkeypatch)
    (data / "data.csv").write_text("placeholder", encoding="utf-8")
    (data / "~$open.xlsx").write_text("lock", encoding="utf-8")
    existing = model.parent / "old candidate"
    existing.mkdir()
    options = collect_options(root=tmp_path, input_fn=answers(
        "wrong", "2", "0", "2", "1", "0", "-5", "one", "7",
        str(model), str(model / "nested"), str(existing), "fresh model"), now=NOW)
    assert not options.multilabel
    assert options.input_path == data / "data.csv"
    assert options.epochs == 7
    assert options.model_dir == model.parent / "fresh model"
    # Файлы блокировки Excel не предлагаются, рабочие файлы остаются неизменными.
    assert "~$open.xlsx" not in capsys.readouterr().out
    assert (model / "model.pth").read_bytes() == b"working model"
    assert not (model / "nested").exists()


# При совпадении времени свободное имя подбирается числовым суффиксом.
def test_proposed_directory_skips_existing_name(tmp_path):
    model = tmp_path / "model"
    first = default_model_dir(model, True, NOW)
    first.mkdir()
    second = default_model_dir(model, True, NOW)
    second.mkdir()
    assert default_model_dir(model, True, NOW).name == first.name + "-3"
    assert default_model_dir(model, False, NOW).name == "model-singlelabel-20261111-153000"


# Пробелы в путях не разбивают аргументы команды обучения.
def test_paths_with_spaces_are_passed_as_whole_arguments(tmp_path, monkeypatch):
    data, _ = configured_paths(tmp_path, monkeypatch)
    (data / "table with spaces.xlsx").write_text("placeholder", encoding="utf-8")
    destination = tmp_path / "custom output" / "my model"
    options = collect_options(root=tmp_path, input_fn=answers("1", "2", "1", "1", str(destination)), now=NOW)
    command = training_command(options, "cpu")
    assert command[command.index("--input") + 1] == str(data / "table with spaces.xlsx")
    assert command[command.index("--model-dir") + 1] == str(destination)
    assert command[command.index("--device") + 1] == "cpu"
    assert "--multilabel" in command
    assert command[command.index("--training-mode") + 1] == "legacy"


# Без таблиц или при завершении ввода обучение и запись кандидата не начинаются.
def test_no_tables_and_cancel_do_not_start_training(tmp_path, monkeypatch, capsys):
    data, model = configured_paths(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="нет таблиц CSV или XLSX"):
        collect_options(root=tmp_path, input_fn=answers(), now=NOW)
    # Даже с доступной таблицей EOF отменяет запуск до вызова train.py.
    (data / "data.csv").write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(config, "TRAIN_DATA_PATH", str(data))
    monkeypatch.setattr(config, "MODEL_PATH", str(model))

    def cancel(prompt):
        raise EOFError

    assert main(["--device", "cpu"], input_fn=cancel) == 1
    assert "Обучение отменено" in capsys.readouterr().out
    assert list(model.parent.iterdir()) == [model]


# Полный диалог запускает настоящий train.py на временной таблице на одну эпоху.
def test_dialogue_starts_one_epoch_with_temporary_data(tmp_path, monkeypatch):
    data = tmp_path / "tables"
    data.mkdir()
    model = tmp_path / "models" / "working"
    source = data / "labeled.csv"
    records = []
    for index in range(6):
        records.extend([
            {"text": f"Стол мебель номер{index}", "label": "home"},
            {"text": f"Ноутбук техника номер{index}", "label": "electronics"},
            {"text": f"Стол и ноутбук вместе номер{index}", "label": "electronics|home"},
        ])
    pd.DataFrame(records).to_csv(source, index=False)
    original = source.read_bytes()
    monkeypatch.setattr(config, "TRAIN_DATA_PATH", str(data))
    monkeypatch.setattr(config, "MODEL_PATH", str(model))
    candidate = model.parent / "dialogue candidate"
    assert main(["--device", "cpu"], input_fn=answers("", "", "", "1", str(candidate))) == 0
    # Сохранённые режим и параметры соответствуют ответам, исходник не изменён.
    vectorizer = joblib.load(candidate / "vectorizer.pkl")
    assert vectorizer.classification_task_ == "multilabel"
    assert vectorizer.training_config_["epochs"] == 1
    assert vectorizer.training_config_["device"] == "cpu"
    assert vectorizer.training_config_["training_mode"] == "best"
    assert vectorizer.training_history_["best_epoch"] == 1
    assert set(json.loads((candidate / "labels_encoded.json").read_text(encoding="utf-8")).values()) == {
        "home", "electronics",
    }
    assert source.read_bytes() == original
    assert not model.exists()
