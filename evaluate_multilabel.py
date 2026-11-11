"""Раздельный подбор порогов и итоговая проверка зафиксированного кандидата."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from models.model import SimpleNN
from utils.dataset_split import prepare_dataset, file_sha256
from utils.multilabel_evaluation import tune_thresholds, quality_report, select_labels


def candidate_hashes(directory):
    return {name: file_sha256(directory / name)
            for name in ("model.pth", "vectorizer.pkl", "labels_encoded.json")}


def evaluate(*, model_dir, input_path, output, stage="tune", selection_path=None,
             objective="f1", min_precision=0.95, min_support=10, batch_size=128):
    directory, output = Path(model_dir), Path(output)
    if output.exists():
        raise ValueError("Каталог отчёта уже существует; укажите новый --output")
    if batch_size < 1:
        raise ValueError("Размер пакета должен быть положительным")
    if stage not in {"tune", "test"}:
        raise ValueError("Этап должен быть tune или test")
    hashes = candidate_hashes(directory)
    vectorizer = joblib.load(directory / "vectorizer.pkl")
    if getattr(vectorizer, "classification_task_", None) != "multilabel":
        raise ValueError("Требуется multilabel-кандидат")
    config = vectorizer.training_config_
    split = prepare_dataset(input_path, seed=config["split_seed"])
    if split.manifest["source_sha256"] != config["source_sha256"]:
        raise ValueError("Исходная таблица отличается от таблицы обучения")
    if (split.manifest["partitions"] != config["partition_records"]
            or split.manifest["version"] != config["split_version"]
            or split.manifest["normalization"] != config["normalization"]):
        raise ValueError("Разделение отличается от использованного при обучении")
    mapping = json.loads((directory / "labels_encoded.json").read_text(encoding="utf-8"))
    labels = [mapping[str(index)] for index in range(len(mapping))]
    if labels != config["classes"]:
        raise ValueError("Порядок категорий не соответствует обучению")
    provenance = {"candidate_sha256": hashes, "source_sha256": split.manifest["source_sha256"],
                  "split_seed": config["split_seed"], "partition_records": config["partition_records"],
                  "labels": labels}
    selection = None
    if stage == "test":
        if selection_path is None:
            raise ValueError("Для test укажите --selection из завершённого этапа tune")
        selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
        if selection.get("provenance") != provenance or selection.get("version") != 1:
            raise ValueError("Пороги относятся к другому кандидату или разделению")
        if selection["policy"].get("comparison") != ">=":
            raise ValueError("Неизвестное правило сравнения порогов")
        select_labels(np.zeros((1, len(labels))), selection["policy"]["thresholds"])
    frame = split.partitions["validation" if stage == "tune" else "test"]
    targets = np.array([[int(label in values) for label in labels] for values in frame.labels])
    model = SimpleNN(len(vectorizer.vocabulary_), 100, len(labels))
    model.load_state_dict(torch.load(directory / "model.pth", map_location="cpu", weights_only=True))
    model.eval()
    vectors = vectorizer.transform(frame.text)
    probabilities = []
    with torch.inference_mode():
        for offset in range(0, len(frame), batch_size):
            batch = torch.from_numpy(vectors[offset:offset + batch_size].toarray().astype(np.float32))
            probabilities.append(model(batch).sigmoid().numpy())
    probabilities = np.concatenate(probabilities)
    if stage == "tune":
        selection = {"version": 1, "provenance": provenance,
                     "policy": tune_thresholds(targets, probabilities, labels, objective=objective,
                                               min_precision=min_precision, min_support=min_support)}
    thresholds = selection["policy"]["thresholds"]
    report = {"stage": stage, "provenance": provenance, "policy": selection["policy"],
              "baseline_0_5": quality_report(targets, probabilities, [0.5] * len(labels), labels),
              "selected": quality_report(targets, probabilities, thresholds, labels),
              "acceptance": "not_decided", "empty_prediction_route": "manual_review_pending_integration"}
    predictions = select_labels(probabilities, thresholds)
    errors = [{"source_record": int(frame.iloc[index]["_source_record"]),
               "text": frame.iloc[index].text, "expected": frame.iloc[index].labels,
               "predicted": [label for label, value in zip(labels, predictions[index]) if value]}
              for index in np.flatnonzero(np.any(predictions != targets, axis=1))]
    if candidate_hashes(directory) != hashes or file_sha256(input_path) != provenance["source_sha256"]:
        raise ValueError("Модель или исходная таблица изменились во время оценки")
    output.mkdir(parents=True, exist_ok=False)
    files = {"report.json": report, "errors.json": errors}
    if stage == "tune":
        files["selection.json"] = selection
    for name, data in files.items():
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Новый каталог локальных отчётов")
    parser.add_argument("--stage", choices=["tune", "test"], default="tune")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--objective", choices=["f1", "precision"])
    parser.add_argument("--min-precision", type=float)
    parser.add_argument("--min-support", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.stage == "test" and any(value is not None for value in
                                    (args.objective, args.min_precision, args.min_support)):
        parser.error("На этапе test параметры порогов берутся только из --selection")
    try:
        report = evaluate(model_dir=args.model_dir, input_path=args.input, output=args.output,
                          stage=args.stage, selection_path=args.selection, objective=args.objective or "f1",
                          min_precision=args.min_precision if args.min_precision is not None else 0.95,
                          min_support=args.min_support if args.min_support is not None else 10,
                          batch_size=args.batch_size)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(report["selected"]["all"], ensure_ascii=False, indent=2))
    print("Пустые результаты направляются на проверку; интеграция маршрута остаётся в следующих чекпоинтах.")


if __name__ == "__main__":
    main()
