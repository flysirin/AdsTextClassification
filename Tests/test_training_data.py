import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from utils.training_data import CATEGORIES, load_training_data


ROOT = Path(__file__).resolve().parents[1]


def record(text="Продаю стол", label="home", **metadata):
    return {"text": text, "label": label, "annotation_status": "completed",
            "review_required": False, "qa_status": "", **metadata}


@pytest.mark.parametrize("ext,sep", [("xlsx", None), ("csv", ","), ("csv", ";")])
def test_preparation_filters_and_normalizes_without_changing_source(tmp_path, ext, sep):
    text = '  Продаю "стол", цена; торг\nДоставка 🪑  '
    records = [
        record(text, " home | home "),
        record(label="home | electronics | electronics", qa_status="DONE"),
        record(label="other|home"),
        record(label="no_ads"),
        record(label="spam"),
        record(label="review", review_required=True),
        record(review_required="TRUE"),
        record(annotation_status="pending"),
        record(qa_status="pending"),
        record(qa_status="running"),
        record(review_required="maybe"),
        record(annotation_status=""),
        record(review_required=""),
        record(label=""),
        record(label="home||electronics"),
        record(label="furniture"),
        record(label="spam|home"),
        record(label="no_ads|spam"),
        record(text=" \n "),
        record(label="review|home"),
        record(text=None, label=None),
    ]
    path = tmp_path / f"training.{ext}"
    source = pd.DataFrame(records)
    if sep:
        source.to_csv(path, sep=sep, index=False, encoding="utf-8-sig")
    else:
        source.to_excel(path, index=False)
    original = path.read_bytes()

    frame, report = load_training_data(path)

    assert path.read_bytes() == original
    assert frame.text[0] == text
    assert frame.label.tolist() == ["home", "electronics|home", "home|other", "no_ads", "spam"]
    assert frame.labels.tolist() == [["home"], ["electronics", "home"], ["home", "other"],
                                     ["no_ads"], ["spam"]]
    assert "label_encoded" not in frame
    assert report.total_rows == 21
    assert report.accepted_rows == 5
    assert report.excluded_rows == 16
    assert report.multilabel_rows == 2
    assert report.exclusion_reasons == {
        "annotation_incomplete": 2, "empty_label": 2, "empty_label_element": 1,
        "empty_text": 2, "exclusive_label_conflict": 2, "invalid_review_required": 2,
        "qa_incomplete": 2, "review_label": 2, "review_required": 2, "unknown_label": 1,
    }
    assert report.examples["review_label"] == [6, 20]
    assert report.examples["empty_label"] == [14, 21]


def test_two_column_input_needs_no_qa_fields(tmp_path):
    path = tmp_path / "simple.csv"
    pd.DataFrame({"text": ["Объявление"] * len(CATEGORIES), "label": sorted(CATEGORIES)}).to_csv(
        path, index=False)
    frame, report = load_training_data(path)
    assert report.accepted_rows == len(CATEGORIES)
    assert report.excluded_rows == 0
    assert {labels[0] for labels in frame.labels} == CATEGORIES


@pytest.mark.parametrize("columns", [{"text": ["Текст"]}, {"label": ["home"]}])
def test_missing_required_columns(tmp_path, columns):
    path = tmp_path / "invalid.csv"
    pd.DataFrame(columns).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Отсутствуют обязательные столбцы"):
        load_training_data(path)


def test_incomplete_annotation_and_unknown_status_fail_closed(tmp_path):
    path = tmp_path / "status.csv"
    pd.DataFrame([
        record(annotation_status="legacy_unverified"),
        record(annotation_status="unexpected"),
        record(qa_status="unexpected"),
        record(review_required=1),
    ]).to_csv(path, index=False)
    frame, report = load_training_data(path)
    assert frame.empty
    assert report.excluded_rows == 4
    assert report.exclusion_reasons == {
        "annotation_incomplete": 2, "invalid_review_required": 1, "qa_incomplete": 1,
    }


def test_report_command_and_training_guard(tmp_path):
    path = tmp_path / "multilabel.csv"
    pd.DataFrame([record(label="electronics|home"), record(label="review", review_required=True)]).to_csv(
        path, index=False)
    original = path.read_bytes()
    result = subprocess.run(
        [sys.executable, "-m", "utils.training_data", "--input", path],
        cwd=ROOT, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["accepted_rows"] == 1
    assert report["excluded_rows"] == 1
    assert report["multilabel_rows"] == 1

    model_dir = tmp_path / "candidate"
    result = subprocess.run(
        [sys.executable, "train.py", "--input", path, "--epochs", "1", "--model-dir", model_dir],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert "BCEWithLogitsLoss" in result.stderr
    assert not model_dir.exists()
    assert path.read_bytes() == original


def test_no_accepted_rows_are_reported_without_writing_model(tmp_path):
    path = tmp_path / "review.csv"
    pd.DataFrame([record(label="review", review_required=True)]).to_csv(path, index=False)
    result = subprocess.run(
        [sys.executable, "-m", "utils.training_data", "--input", path],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["accepted_rows"] == 0
    assert "Нет строк" in result.stderr
    result = subprocess.run(
        [sys.executable, "train.py", "--input", path, "--model-dir", tmp_path / "candidate"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Нет строк" in result.stderr
    assert not (tmp_path / "candidate").exists()
