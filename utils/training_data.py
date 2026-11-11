"""Проверка и подготовка полной многометочной разметки CSV/XLSX."""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from utils.tables import read_table


CATEGORIES = frozenset({
    "animals", "children", "clothing", "electronics", "food", "health",
    "hobby", "home", "jobs", "no_ads", "other", "real_estate", "services",
    "spam", "transport",
})
EXCLUSIVE_CATEGORIES = frozenset({"no_ads", "spam"})


@dataclass
class PreparationReport:
    total_rows: int
    accepted_rows: int
    excluded_rows: int
    multilabel_rows: int
    exclusion_reasons: dict[str, int]
    examples: dict[str, list[int]]


def _string(value):
    return "" if pd.isna(value) else str(value).strip()


def load_training_data(file_path, label_name="label"):
    """Вернуть допущенные строки с колонкой labels и отчёт без записи на диск.

    Номера в examples — номера записей с 1, без заголовка, не строки файла.
    Причины исключения могут пересекаться; examples содержит первые 5 номеров.
    """
    frame = read_table(file_path)
    missing = {"text", label_name} - set(frame.columns)
    if missing:
        raise ValueError("Отсутствуют обязательные столбцы: " + ", ".join(sorted(missing)))

    accepted = []
    parsed_labels = []
    counts = Counter()
    examples = defaultdict(list)
    for position, values in enumerate(frame.itertuples(index=False, name=None)):
        row = dict(zip(frame.columns, values))
        reasons = set()
        text = _string(row["text"])
        label = _string(row[label_name])
        parts = [part.strip() for part in label.split("|")]
        labels = sorted(set(parts) - {""})
        if not text:
            reasons.add("empty_text")
        if not label:
            reasons.add("empty_label")
        elif "" in parts:
            reasons.add("empty_label_element")
        if "review" in labels:
            reasons.add("review_label")
        if set(labels) - CATEGORIES - {"review"}:
            reasons.add("unknown_label")
        if EXCLUSIVE_CATEGORIES.intersection(labels) and len(labels) > 1:
            reasons.add("exclusive_label_conflict")

        if "review_required" in row:
            review = _string(row["review_required"]).casefold()
            if review == "true":
                reasons.add("review_required")
            elif review != "false":
                reasons.add("invalid_review_required")
        if "annotation_status" in row:
            if _string(row["annotation_status"]).casefold() != "completed":
                reasons.add("annotation_incomplete")
        if "qa_status" in row:
            if _string(row["qa_status"]).casefold() not in {"", "done"}:
                reasons.add("qa_incomplete")

        if reasons:
            for reason in sorted(reasons):
                counts[reason] += 1
                if len(examples[reason]) < 5:
                    examples[reason].append(position + 1)
        else:
            accepted.append(position)
            parsed_labels.append(labels)

    prepared = frame.iloc[accepted].copy().reset_index(drop=True)
    # Сохраняем текст без нормализации; нормализуем только представление меток.
    prepared["text"] = prepared["text"].astype(str)
    prepared[label_name] = ["|".join(labels) for labels in parsed_labels]
    prepared["labels"] = parsed_labels
    prepared["_source_record"] = [position + 1 for position in accepted]
    report = PreparationReport(
        total_rows=len(frame),
        accepted_rows=len(prepared),
        excluded_rows=len(frame) - len(prepared),
        multilabel_rows=sum(len(labels) > 1 for labels in parsed_labels),
        exclusion_reasons=dict(sorted(counts.items())),
        examples=dict(sorted(examples.items())),
    )
    return prepared, report


def main():
    parser = argparse.ArgumentParser(description="Проверить разметку без обучения и изменения таблицы")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    try:
        _, report = load_training_data(args.input)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    if report.accepted_rows == 0:
        parser.exit(1, "Нет строк, допущенных к обучению.\n")


if __name__ == "__main__":
    main()
