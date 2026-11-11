"""Выбор параметров обучения с каталогами из config/config.py."""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from config import config
from utils.training import DEFAULT_EPOCHS, DEFAULT_PATIENCE, DEFAULT_MIN_DELTA


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainingOptions:
    multilabel: bool
    input_path: Path
    epochs: int
    model_dir: Path
    training_mode: str = "legacy"


def resolve_path(path, root):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def display_path(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def default_model_dir(model_path, multilabel, now=None):
    now = now or datetime.now(ZoneInfo("Asia/Tbilisi"))
    task = "multilabel" if multilabel else "singlelabel"
    base = model_path.with_name(f"{model_path.name}-{task}-{now:%Y%m%d-%H%M%S}")
    candidate = base
    number = 2
    while candidate.exists() or candidate.is_symlink():
        candidate = base.with_name(f"{base.name}-{number}")
        number += 1
    return candidate


def collect_options(*, root=ROOT, input_fn=None, now=None):
    ask = input_fn or input
    root = Path(root).resolve()
    data_path = resolve_path(config.TRAIN_DATA_PATH, root)
    model_path = resolve_path(config.MODEL_PATH, root)
    files = sorted((path for path in data_path.glob("*") if path.is_file()
                    and path.suffix.lower() in {".csv", ".xlsx"} and not path.name.startswith("~$")),
                   key=lambda path: (path.name.casefold(), path.name))
    if not files:
        raise ValueError(f"В {data_path} нет таблиц CSV или XLSX")

    print("Режим обучения")
    print("1. Multilabel — одна или несколько категорий")
    print("2. Singlelabel — только одна категория")
    while True:
        mode = ask("Выберите режим [1]: ").strip() or "1"
        if mode in {"1", "2"}:
            break
        print("Введите 1 или 2")
    multilabel = mode == "1"
    training_mode = "legacy"
    if multilabel:
        print("\nСпособ обучения")
        print("1. Новый: лучшие веса по validation и ранняя остановка")
        print("2. Старый: все эпохи и последние веса")
        while True:
            choice = ask("Выберите способ [1]: ").strip() or "1"
            if choice in {"1", "2"}:
                training_mode = "best" if choice == "1" else "legacy"
                break
            print("Введите 1 или 2")
        if training_mode == "best":
            print(f"Остановка: patience={DEFAULT_PATIENCE}, min_delta={DEFAULT_MIN_DELTA}")

    print(f"\nТаблицы из {display_path(data_path, root)}")
    for number, path in enumerate(files, 1):
        print(f"{number}. {path.name}")
    default_index = next((index for index, path in enumerate(files, 1)
                          if path.name.casefold() == "labeled.csv"), 1)
    while True:
        selection = ask(f"Номер таблицы [{default_index}]: ").strip() or str(default_index)
        if selection.isdecimal() and 1 <= int(selection) <= len(files):
            source = files[int(selection) - 1]
            break
        print(f"Введите номер от 1 до {len(files)}")

    while True:
        value = ask(f"Максимальное количество эпох [{DEFAULT_EPOCHS}]: ").strip() or str(DEFAULT_EPOCHS)
        if value.isdecimal() and int(value) > 0:
            epochs = int(value)
            break
        print("Количество эпох должно быть целым числом больше нуля")

    proposed = default_model_dir(model_path, multilabel, now)
    while True:
        value = ask(f"Каталог сохранения [{display_path(proposed, root)}]: ").strip()
        if not value:
            candidate = proposed
        else:
            path = Path(value).expanduser()
            # Одно имя означает новый каталог рядом с настроенным MODEL_PATH.
            if not path.is_absolute() and len(path.parts) == 1 and value not in {".", ".."}:
                path = model_path.parent / path
            candidate = resolve_path(path, root)
        if candidate == model_path or model_path in candidate.parents:
            print("Выберите новый каталог рядом с рабочей моделью, а не внутри неё")
        elif candidate.exists() or candidate.is_symlink():
            print("Этот путь уже существует. Введите другое имя")
        else:
            return TrainingOptions(multilabel, source, epochs, candidate, training_mode)


def training_command(options, device):
    command = [sys.executable, str(ROOT / "train.py"), "--input", str(options.input_path),
               "--epochs", str(options.epochs), "--model-dir", str(options.model_dir), "--device", device]
    if options.multilabel:
        command.append("--multilabel")
    command.extend(["--training-mode", options.training_mode])
    return command


def main(argv=None, *, input_fn=None):
    parser = argparse.ArgumentParser(description="Выбрать режим, таблицу, эпохи и каталог модели")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args(argv)
    try:
        options = collect_options(input_fn=input_fn)
    except ValueError as error:
        parser.error(str(error))
    except EOFError:
        print("\nОбучение отменено: ввод завершён")
        return 1
    except KeyboardInterrupt:
        print("\nОбучение отменено")
        return 130
    print("\nПараметры обучения")
    print("Режим: " + ("multilabel" if options.multilabel else "singlelabel"))
    print(f"Таблица: {display_path(options.input_path, ROOT)}")
    print(f"Эпохи: {options.epochs}")
    print(f"Способ обучения: {options.training_mode}")
    print(f"Каталог модели: {display_path(options.model_dir, ROOT)}")
    print("Запускаю обучение\n", flush=True)
    return subprocess.run(training_command(options, args.device), cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
