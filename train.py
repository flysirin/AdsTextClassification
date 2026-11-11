import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import config
from utils.dataset_split import file_sha256, split_training_data
from utils.training import DEFAULT_EPOCHS, DEFAULT_PATIENCE, DEFAULT_MIN_DELTA, save_model, train_model
from utils.util import get_input_data_file_excel, prepared_data_excel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Максимальное число эпох")
    parser.add_argument("--training-mode", choices=["legacy", "best"], default="legacy",
                        help="legacy: последние веса; best: лучшие веса по validation (multilabel)")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--no-early-stopping", action="store_true", help="Все эпохи, сохраняя лучшие веса в best")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--multilabel", action="store_true", help="Обучать независимые категории с BCEWithLogitsLoss")
    parser.add_argument("--pos-weight", action="store_true", help="Веса категорий из train для multilabel")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Число эпох и размер пакета должны быть больше нуля")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed должен быть от 0 до 2**32 - 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA недоступна для текущего окружения или драйвера")
    if args.pos_weight and not args.multilabel:
        parser.error("--pos-weight требует --multilabel")
    directory = args.model_dir or Path(config.MODEL_PATH)
    if args.multilabel:
        current = Path(config.MODEL_PATH).resolve()
        candidate = directory.resolve()
        if args.model_dir is None or candidate == current or current in candidate.parents:
            parser.error("Для --multilabel укажите --model-dir с новым каталогом кандидата вне saved/model")
        if directory.exists():
            parser.error("Каталог кандидата уже существует; укажите новый --model-dir")
    try:
        source = args.input or get_input_data_file_excel()
        digest = file_sha256(source)
        frame = prepared_data_excel(source)
        if not args.multilabel and frame["labels"].map(len).gt(1).any():
            parser.error("Для нескольких категорий используйте --multilabel (BCEWithLogitsLoss) "
                         "и --model-dir с новым каталогом кандидата")
        split = split_training_data(frame, seed=args.seed)
        if file_sha256(source) != digest:
            raise ValueError("Исходный файл изменился во время подготовки; повторите обучение")
        print("Разделение данных: " + json.dumps(split.report, ensure_ascii=False))
        result = train_model(split, multilabel=args.multilabel, epochs=args.epochs, device=args.device,
                             seed=args.seed, batch_size=args.batch_size, pos_weight=args.pos_weight,
                             source_sha256=digest, verbose=True, training_mode=args.training_mode,
                             patience=args.patience, min_delta=args.min_delta,
                             early_stopping=not args.no_early_stopping)
        history = result.vectorizer.training_history_
        print(f"Эпох выполнено: {history['epochs_completed']}/{args.epochs}; "
              f"причина завершения: {history['stop_reason']}")
        if history["best_epoch"] is not None:
            print(f"Лучшая эпоха: {history['best_epoch']}; validation BCE={result.validation_loss:.6f}")
        if args.multilabel:
            print(f"Validation BCE={result.validation_loss:.6f}; независимый test не использован")
        else:
            test = split.partitions["test"]
            predictions = []
            vectors = result.vectorizer.transform(test.text)
            with torch.inference_mode():
                for offset in range(0, len(test), args.batch_size):
                    batch = torch.tensor(vectors[offset:offset + args.batch_size].toarray(), dtype=torch.float32)
                    predictions.extend(result.model(batch).argmax(dim=1).tolist())
            indices = {label: index for index, label in result.labels.items()}
            y = [indices[labels[0]] for labels in test.labels]
            p, r, f, _ = precision_recall_fscore_support(y, predictions, average="weighted", zero_division=0)
            print(f"Accuracy={accuracy_score(y, predictions):.4f} Precision={p:.4f} Recall={r:.4f} F1={f:.4f}")
        save_model(result, directory, allow_existing=not args.multilabel)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
