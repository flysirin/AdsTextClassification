import json

import numpy as np
import pandas as pd
import pytest

import evaluate_multilabel as command
from utils.dataset_split import prepare_dataset, file_sha256
from utils.multilabel_evaluation import tune_thresholds, select_labels, quality_report
from utils.training import train_model, save_model


# CLI фиксирует настройки test и корректно передаёт значения по умолчанию в tune.
def test_cli_defaults_and_test_rejects_tuning(monkeypatch, capsys):
    import sys
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"selected": {"all": {"rows": 1}}}

    monkeypatch.setattr(command, "evaluate", fake_evaluate)
    arguments = ["evaluate_multilabel.py", "--model-dir", "candidate", "--input", "data.csv", "--output", "report"]
    monkeypatch.setattr(sys, "argv", arguments)
    command.main()
    assert captured["objective"] == "f1"
    assert captured["min_support"] == 10
    # Синтаксис --objective=value также не позволяет менять пороги на test.
    monkeypatch.setattr(sys, "argv", arguments + ["--stage", "test", "--objective=f1"])
    with pytest.raises(SystemExit) as error:
        command.main()
    assert error.value.code == 2
    assert "только из --selection" in capsys.readouterr().err


# Пороги выбираются воспроизводимо, редкие метки получают общий порог.
def test_thresholds_f1_rare_and_empty_predictions():
    targets = np.array([[1, 0], [1, 1], [0, 0], [0, 0]])
    probabilities = np.array([[0.4, 0.1], [0.35, 0.45], [0.2, 0.3], [0.1, 0.2]])
    result = tune_thresholds(targets, probabilities, ["home", "electronics"], min_support=2)
    assert result == tune_thresholds(targets, probabilities, ["home", "electronics"], min_support=2)
    assert result["decisions"][1]["rule"] == "low_support"
    assert result["thresholds"][1] == result["common_threshold"]
    # Сравнение включает границу и не заменяет пустой набор на argmax.
    assert select_labels([[0.4, 0.1], [0.0, 0.0]], [0.4, 0.2]).tolist() == [[1, 0], [0, 0]]
    report = quality_report(targets, probabilities, result["thresholds"], ["home", "electronics"])
    assert report["all"]["rows"] == 4
    assert report["singlelabel"]["rows"] == 1
    assert report["multilabel"]["rows"] == 1
    assert report["all"]["per_category"]["electronics"]["support"] == 1


# Недостижимая precision не выдаётся за успешный подбор, плохие оценки отклоняются.
def test_precision_and_invalid_inputs():
    y = np.array([[1], [0]])
    with pytest.raises(ValueError, match="нет общего порога"):
        tune_thresholds(y, np.array([[0.1], [0.9]]), ["home"], objective="precision")
    result = tune_thresholds(y, np.array([[0.8], [0.1]]), ["home"], objective="precision", min_support=1)
    assert select_labels([[0.8], [0.1]], result["thresholds"]).tolist() == [[1], [0]]
    # Проверяем числовые ограничения до вычисления метрик.
    for scores in ([[float("nan")]], [[1.1]]):
        with pytest.raises(ValueError):
            select_labels(scores, [0.5])
    with pytest.raises(ValueError):
        tune_thresholds(y, np.array([[0.8], [0.1]]), ["home"], min_support=0)


# CSV и XLSX проходят два этапа без изменения кандидата, данных и состава частей.
@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_tune_then_fixed_test_and_provenance(tmp_path, monkeypatch, extension):
    source = tmp_path / f"data.{extension}"
    frame = pd.DataFrame([{"text": f"{word} пример{index}", "label": label}
                          for index in range(10) for word, label in
                          [("диван", "home"), ("ноутбук", "electronics"),
                           ("диван ноутбук", "home|electronics")]])
    if extension == "csv":
        frame.to_csv(source, index=False)
    else:
        frame.to_excel(source, index=False)
    split = prepare_dataset(source)
    model_dir = tmp_path / "candidate"
    result = train_model(split, multilabel=True, training_mode="best", epochs=2,
                         source_sha256=file_sha256(source))
    save_model(result, model_dir)
    hashes = command.candidate_hashes(model_dir)
    source_hash = file_sha256(source)
    # Подбор не обращается к test; итоговая оценка не подбирает пороги повторно.
    original_test = split.partitions["test"]
    split.partitions["test"] = None
    monkeypatch.setattr(command, "prepare_dataset", lambda *args, **kwargs: split)
    tuned = command.evaluate(model_dir=model_dir, input_path=source, output=tmp_path / "tune")
    assert tuned["stage"] == "tune"
    selection = tmp_path / "tune/selection.json"
    split.partitions["test"] = original_test
    split.partitions["validation"] = None
    monkeypatch.setattr(command, "tune_thresholds", lambda *args, **kwargs: pytest.fail("Повторный подбор на test"))
    tested = command.evaluate(model_dir=model_dir, input_path=source, output=tmp_path / "test",
                              stage="test", selection_path=selection)
    assert tested["policy"] == tuned["policy"]
    assert tested["selected"]["all"]["rows"] == len(original_test)
    assert command.candidate_hashes(model_dir) == hashes
    assert file_sha256(source) == source_hash
    assert not (tmp_path / "test/selection.json").exists()
    # Отчёты не перезаписываются, чужие пороги и изменённый источник отвергаются.
    with pytest.raises(ValueError, match="уже существует"):
        command.evaluate(model_dir=model_dir, input_path=source, output=tmp_path / "test")
    payload = json.loads(selection.read_text())
    payload["provenance"]["candidate_sha256"]["model.pth"] = "wrong"
    selection.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="другому кандидату"):
        command.evaluate(model_dir=model_dir, input_path=source, output=tmp_path / "wrong",
                         stage="test", selection_path=selection)
    with pytest.raises(ValueError, match="укажите --selection"):
        command.evaluate(model_dir=model_dir, input_path=source, output=tmp_path / "missing", stage="test")
    assert not (tmp_path / "wrong").exists()
