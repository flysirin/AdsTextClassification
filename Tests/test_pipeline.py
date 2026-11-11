import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from AppFlaskPyTorch.app import create_app
from utils.tables import read_table
from utils.dataset_split import split_training_data
from utils.training_data import load_training_data

ROOT = Path(__file__).resolve().parents[1]


# Сквозная совместимость обучения, HTTP и пакетного прогноза для CSV и XLSX.
@pytest.mark.parametrize("ext,sep", [("xlsx", None), ("csv", ","), ("csv", ";")])
def test_train_predict_and_http(tmp_path, ext, sep):
    path = tmp_path / f"sample.{ext}"
    frame = pd.DataFrame({"text": [f'Продаю "стол", дуб; цена\n100 товарномер{i}' for i in range(10)]
                                   + [f"Ищу работу поваром вакансияномер{i}" for i in range(10)],
                          "label": ["home"] * 10 + ["jobs"] * 10})
    frame = pd.concat([frame, pd.DataFrame({"text": ["Нужен контекст"], "label": ["review"]})],
                      ignore_index=True)
    if sep:
        frame.to_csv(path, index=False, sep=sep, encoding="utf-8-sig")
    else:
        frame.to_excel(path, index=False)
    # Одна эпоха создаёт временный комплект без служебной категории review.
    model = tmp_path / "model"
    subprocess.run([sys.executable, "train.py", "--input", path, "--model-dir", model, "--epochs", "1"],
                   cwd=ROOT, check=True)
    assert set(json.loads((model / "labels_encoded.json").read_text(encoding="utf-8")).values()) == {"home", "jobs"}
    # Слова только из валидации и теста не должны расширять обучающий словарь.
    prepared, _ = load_training_data(path)
    split = split_training_data(prepared)
    analyzer = CountVectorizer().build_analyzer()
    training_tokens = {token for text in split.partitions["train"].text for token in analyzer(text)}
    heldout_tokens = {token for part in ("validation", "test")
                      for text in split.partitions[part].text for token in analyzer(text)} - training_tokens
    vocabulary = set(joblib.load(model / "vectorizer.pkl").vocabulary_)
    assert heldout_tokens
    assert vocabulary == training_tokens
    assert not vocabulary & heldout_tokens
    # Один комплект даёт одинаковую категорию через API и оба формата таблиц.
    api = create_app(model).test_client()
    assert api.get("/health").json["model_ready"]
    expected = api.post("/predict", json={"messages": {"id": frame.text[0]}}).json["predictions"]["id"]
    for output_ext in ("csv", "xlsx"):
        output = tmp_path / f"prediction.{output_ext}"
        subprocess.run([sys.executable, "run_predict_data.py", "--input", path, "--model-dir", model,
                        "--output", output], cwd=ROOT, check=True)
        result = read_table(output)
        assert result.text.tolist() == frame.text.tolist()
        assert result.label[0] == expected


# Работающий сервис без модели не объявляет готовность и возвращает 503 на прогноз.
def test_missing_model(tmp_path):
    client = create_app(tmp_path).test_client()
    assert not client.get("/health").json["model_ready"]
    assert client.post("/predict", json={"messages": {"1": "test"}}).status_code == 503


# В одноколоночном CSV запятые и точки с запятой остаются частью объявления.
def test_single_column_csv(tmp_path):
    path = tmp_path / "messages.csv"
    path.write_text('text\nпродам стол, срочно; торг\n', encoding="utf-8-sig")
    assert read_table(path).text.tolist() == ["продам стол, срочно; торг"]
