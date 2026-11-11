import json
from dataclasses import asdict
from pathlib import Path

from config import config
from utils.training_data import load_training_data


def prepared_data_excel(file_path, label_name="label"):
    frame, report = load_training_data(file_path, label_name)
    print("Подготовка данных: " + json.dumps(asdict(report), ensure_ascii=False))
    if frame.empty:
        raise ValueError("Нет строк, допущенных к обучению")
    return frame


def get_input_data_file_excel(dir_path=config.TRAIN_DATA_PATH, label_name="label"):
    files = sorted(p for p in Path(dir_path).glob("*") if p.is_file() and p.suffix.lower() in {".xlsx", ".csv"})
    if not files:
        raise FileNotFoundError(f"В {dir_path} нет файлов .xlsx или .csv")
    for number, path in enumerate(files, 1):
        print(f"{number}. {path.name}")
    selection = input("Номер файла: ").strip()
    if not selection.isdecimal() or not 1 <= int(selection) <= len(files):
        raise ValueError("Неверный номер файла")
    return files[int(selection) - 1]


def get_label_dict():
    data = json.loads((Path(config.MODEL_PATH) / "labels_encoded.json").read_text(encoding="utf-8"))
    return {int(key): value for key, value in data.items()}
