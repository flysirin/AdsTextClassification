import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from utils.dataset_split import normalize_text, prepare_dataset, save_split, split_training_data
from utils.training_data import load_training_data


ROOT = Path(__file__).resolve().parents[1]


# Синтетические независимые объявления с одиночной и совместной разметкой.
def samples():
    return [{"message_id": f"ad-{index}", "text": f"Объявление номер {index}",
             "label": "home|electronics" if index % 3 == 0 else "home" if index % 3 == 1 else "jobs"}
            for index in range(90)]


# Нормализация объединяет Unicode, регистр и пробелы, но сохраняет пунктуацию.
def test_normalization():
    assert normalize_text("  ПРоДаю\u00a0Ａ  \nСтол  ") == "продаю a стол"
    assert normalize_text("стол!") != normalize_text("стол")


# Связанные ID и тексты образуют общую группу; конфликты и повторы исключаются.
def test_grouping_conflicts_duplicates_and_transitive_ids(tmp_path):
    records = samples() + [
        {"message_id": "same-id", "text": "Первый связанный текст", "label": "home"},
        {"message_id": "same-id", "text": "Другой связанный текст", "label": "home"},
        {"message_id": "third-id", "text": "  ДРУГОЙ   СВЯЗАННЫЙ\nТЕКСТ ", "label": "home"},
        {"message_id": "third-id", "text": "Третий связанный текст", "label": "home"},
        {"message_id": "conflict", "text": "Спорный текст", "label": "home"},
        {"message_id": "other-conflict", "text": " СПОРНЫЙ  ТЕКСТ ", "label": "electronics"},
        {"message_id": "other-conflict", "text": "Ещё один спорный текст", "label": "electronics"},
        {"message_id": "duplicate", "text": "Объявление номер 1", "label": "home"},
    ]
    path = tmp_path / "data.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    frame, _ = load_training_data(path)
    original = frame.copy(deep=True)
    result = split_training_data(frame)
    # Исходная таблица неизменна, а отчёт объясняет все исключённые записи.
    pd.testing.assert_frame_equal(frame, original)
    assert result.report["conflict_groups"] == 1
    assert result.report["conflict_rows"] == 3
    assert result.report["duplicate_rows"] == 2
    assert result.report["retained_rows"] == 93
    assert result.manifest["conflicts"][0]["records"] == [95, 96, 97]
    assert result.manifest["duplicates"] == [{"record": 93, "kept_record": 92},
                                             {"record": 98, "kept_record": 2}]
    # Транзитивная цепочка ID и текстов остаётся целиком в одной части.
    linked = next(group for group in result.manifest["groups"] if 91 in group["records"])
    assert linked["records"] == [91, 92, 93, 94]
    assert linked["kept_records"] == [91, 92, 94]
    # Между частями нет пересечений или повторного учёта сохранённых записей.
    assert all(count == 0 for pair in result.report["overlaps"].values() for count in pair.values())
    records_by_part = [set(values) for values in result.manifest["partitions"].values()]
    assert len(set.union(*records_by_part)) == 93
    assert sum(len(values) for values in records_by_part) == 93
    assert not {95, 96, 97, 93, 98} & set.union(*records_by_part)
    assert any({91, 92, 94}.issubset(values) for values in records_by_part)


# Seed воспроизводит разделение, а единственный редкий пример остаётся в train.
def test_reproducibility_and_rare_categories(tmp_path):
    records = samples() + [{"message_id": "rare", "text": "Единственный редкий пример", "label": "health"}]
    path = tmp_path / "data.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    first = prepare_dataset(path, seed=42)
    second = prepare_dataset(path, seed=42)
    assert first.manifest == second.manifest
    assert first.report == second.report
    assert first.manifest["partitions"] != prepare_dataset(path, seed=43).manifest["partitions"]
    # Отсутствие редкой категории в контрольных частях явно отражено в отчёте.
    assert first.report["category_support"]["health"] == {
        "train": 1, "validation": 0, "test": 0, "independent_groups": 1,
    }
    assert first.report["missing_categories"]["health"] == ["validation", "test"]
    assert "health" in first.report["low_evaluation_support"]
    for label in ("home", "electronics", "jobs"):
        assert all(first.report["category_support"][label][part] > 0 for part in ("train", "validation", "test"))
    assert first.report["partitions"]["train"]["rows"] > first.report["partitions"]["test"]["rows"]


# Во всех форматах манифест ссылается на исходные записи после фильтрации review.
@pytest.mark.parametrize("ext,sep", [("csv", ","), ("csv", ";"), ("xlsx", None)])
def test_persisted_composition_and_source_references(tmp_path, ext, sep):
    source = tmp_path / f"input.{ext}"
    records = [{"text": "Не размечено", "label": "review"}] + samples()
    if sep:
        pd.DataFrame(records).to_csv(source, sep=sep, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(records).to_excel(source, index=False)
    original = source.read_bytes()
    result = prepare_dataset(source)
    output = tmp_path / "split"
    save_split(result, output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["preparation"]["excluded_rows"] == 1
    assert manifest == result.manifest
    assert report == result.report
    assert len(manifest["source_sha256"]) == 64
    # Номера соответствуют исходным CSV/XLSX, а не новой нумерации после фильтрации.
    for part, part_records in manifest["partitions"].items():
        assert part_records == result.partitions[part]["_source_record"].tolist()
        assert min(part_records) >= 2
        assert result.partitions[part].text.tolist() == [records[number - 1]["text"] for number in part_records]
    # Тексты не попадают в манифест; повторная запись не повреждает файлы.
    assert "Объявление номер" not in (output / "manifest.json").read_text(encoding="utf-8")
    snapshot = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        save_split(result, output)
    assert (output / "manifest.json").read_bytes() == snapshot
    assert source.read_bytes() == original


# Команда подготовки создаёт только манифест и отчёт без запуска модели.
def test_command_creates_manifest_without_training(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame(samples()).to_csv(path, index=False)
    output = tmp_path / "output"
    command = [sys.executable, "-m", "utils.dataset_split", "--input", path, "--output", output]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["retained_rows"] == 90
    assert sorted(file.name for file in output.iterdir()) == ["manifest.json", "report.json"]
    # Существующий каталог вывода нельзя перезаписать повторной командой.
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2


# Повторные строки не заменяют три независимые группы для разделения.
def test_too_few_independent_groups_fail_clearly(tmp_path):
    path = tmp_path / "small.csv"
    pd.DataFrame({"text": ["Первый"] * 10 + ["Второй"] * 10,
                  "label": ["home"] * 10 + ["jobs"] * 10}).to_csv(path, index=False)
    frame, _ = load_training_data(path)
    with pytest.raises(ValueError, match="три независимые группы"):
        split_training_data(frame)
