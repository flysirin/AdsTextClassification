"""Воспроизводимое групповое разделение разметки без утечки повторов."""

import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd

from utils.training_data import CATEGORIES, load_training_data


RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
NORMALIZATION = "nfkc-casefold-whitespace-v1"


@dataclass
class DatasetSplit:
    partitions: dict[str, pd.DataFrame]
    report: dict
    manifest: dict


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def file_sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def split_training_data(frame, seed=42):
    """Группировать связанные записи, убрать конфликты и повторы, разделить.

    На входе — результат load_training_data, включая _source_record.
    Исходный DataFrame не изменяется. Пропорции приблизительны: группа неделима.
    """
    needed = {"text", "labels", "_source_record"}
    if not needed.issubset(frame.columns):
        raise ValueError("Для разделения сначала вызовите load_training_data")
    frame = frame.reset_index(drop=True)
    texts = [normalize_text(text) for text in frame.text]
    ids = ["" if pd.isna(value) else str(value).strip()
           for value in frame.get("message_id", pd.Series("", index=frame.index))]
    source_records = [int(value) for value in frame["_source_record"]]
    if len(set(source_records)) != len(source_records):
        raise ValueError("Номера исходных записей должны быть уникальны")

    parents = list(range(len(frame)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen_texts, seen_ids = {}, {}
    for index, (text, message_id) in enumerate(zip(texts, ids)):
        for value, seen in ((text, seen_texts), (message_id, seen_ids)):
            if value:
                if value in seen:
                    parents[root(index)] = root(seen[value])
                else:
                    seen[value] = index
    components = defaultdict(list)
    for index in range(len(frame)):
        components[root(index)].append(index)

    groups, conflicts, duplicates = [], [], []
    for indices in components.values():
        signatures = sorted({tuple(frame.labels.iloc[index]) for index in indices})
        identity = {"texts": sorted({texts[index] for index in indices}),
                    "message_ids": sorted({ids[index] for index in indices if ids[index]})}
        group_id = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        records = [source_records[index] for index in indices]
        if len(signatures) != 1:
            conflicts.append({"group_id": group_id, "records": records,
                              "label_sets": [list(labels) for labels in signatures]})
            continue
        representatives = {}
        kept = []
        for index in indices:
            if texts[index] in representatives:
                duplicates.append({"record": source_records[index],
                                   "kept_record": source_records[representatives[texts[index]]]})
            else:
                representatives[texts[index]] = index
                kept.append(index)
        groups.append({"group_id": group_id, "indices": indices, "kept": kept,
                       "records": records, "labels": list(signatures[0])})

    if len(groups) < 3:
        raise ValueError("После исключения конфликтов и повторов нужны хотя бы три независимые группы "
                         "для train/validation/test")
    group_support = Counter(label for group in groups for label in group["labels"])
    remaining = group_support.copy()
    total_support = Counter()
    for group in groups:
        for label in group["labels"]:
            total_support[label] += len(group["kept"])
    total_rows = sum(len(group["kept"]) for group in groups)
    rng = random.Random(seed)
    # Сначала редкие категории; случайный приоритет воспроизводим для данного seed.
    groups.sort(key=lambda group: group["group_id"])
    priorities = {group["group_id"]: rng.random() for group in groups}
    groups.sort(key=lambda group: (min(group_support[label] for label in group["labels"]),
                                   -len(group["labels"]), priorities[group["group_id"]]))
    rows = Counter()
    support = {part: Counter() for part in RATIOS}
    covered = {label: set() for label in group_support}
    for group in groups:
        labels = group["labels"]
        must_train = any(remaining[label] == 1 and "train" not in covered[label] for label in labels)

        def score(part):
            urgent = sum(1 / group_support[label] for label in labels
                         if group_support[label] >= 3
                         and remaining[label] <= len(RATIOS) - len(covered[label])
                         and part not in covered[label])
            deficit = sum((total_support[label] * RATIOS[part] - support[part][label])
                          / max(total_support[label] * RATIOS[part], 1) for label in labels) / len(labels)
            capacity = total_rows * RATIOS[part] - rows[part]
            return urgent, deficit, capacity / (total_rows * RATIOS[part]), capacity

        part = "train" if must_train else max(RATIOS, key=score)
        group["partition"] = part
        rows[part] += len(group["kept"])
        for label in labels:
            support[part][label] += len(group["kept"])
            covered[label].add(part)
            remaining[label] -= 1

    if any(rows[part] == 0 for part in RATIOS):
        raise ValueError("Невозможно получить три непустые части с обучающими примерами каждой категории: "
                         "добавьте независимые группы, особенно для редких меток")
    partitions = {}
    keys = {part: {"groups": set(), "texts": set(), "message_ids": set()} for part in RATIOS}
    for part in RATIOS:
        selected = [group for group in groups if group["partition"] == part]
        indices = sorted(index for group in selected for index in group["kept"])
        partitions[part] = frame.iloc[indices].copy().reset_index(drop=True)
        keys[part]["groups"] = {group["group_id"] for group in selected}
        all_indices = [index for group in selected for index in group["indices"]]
        keys[part]["texts"] = {texts[index] for index in all_indices}
        keys[part]["message_ids"] = {ids[index] for index in all_indices if ids[index]}
    overlaps = {f"{left}/{right}": {key: len(keys[left][key] & keys[right][key]) for key in keys[left]}
                for left, right in combinations(RATIOS, 2)}
    if any(count for pair in overlaps.values() for count in pair.values()):
        raise RuntimeError("Обнаружено пересечение групп между частями")
    category_support = {label: {**{part: support[part][label] for part in RATIOS},
                                "independent_groups": group_support[label]}
                        for label in sorted(CATEGORIES)}
    report = {
        "seed": seed, "ratios": RATIOS, "normalization": NORMALIZATION,
        "accepted_rows": len(frame), "conflict_groups": len(conflicts),
        "conflict_rows": sum(len(group["records"]) for group in conflicts),
        "duplicate_rows": len(duplicates), "retained_rows": total_rows, "retained_groups": len(groups),
        "partitions": {part: {"rows": rows[part], "groups": len(keys[part]["groups"])} for part in RATIOS},
        "category_support": category_support, "overlaps": overlaps,
        "missing_categories": {label: [part for part in RATIOS if support[part][label] == 0]
                               for label in sorted(group_support)
                               if any(support[part][label] == 0 for part in RATIOS)},
        "low_evaluation_support": [label for label in sorted(group_support)
                                   if min(support["validation"][label], support["test"][label]) < 10],
    }
    manifest = {
        "version": 1, "seed": seed, "ratios": RATIOS, "normalization": NORMALIZATION,
        "partitions": {part: [int(value) for value in partitions[part]["_source_record"]] for part in RATIOS},
        "groups": [{"group_id": group["group_id"], "records": group["records"],
                    "kept_records": [source_records[index] for index in group["kept"]],
                    "labels": group["labels"], "partition": group["partition"]}
                   for group in sorted(groups, key=lambda group: group["group_id"])],
        "conflicts": sorted(conflicts, key=lambda group: group["group_id"]),
        "duplicates": sorted(duplicates, key=lambda duplicate: duplicate["record"]),
    }
    return DatasetSplit(partitions, report, manifest)


def prepare_dataset(path, seed=42):
    digest = file_sha256(path)
    frame, preparation = load_training_data(path)
    result = split_training_data(frame, seed)
    if file_sha256(path) != digest:
        raise ValueError("Исходный файл изменился во время подготовки; повторите операцию")
    result.manifest["source_sha256"] = digest
    result.report["source_sha256"] = digest
    result.report["preparation"] = asdict(preparation)
    return result


def save_split(result, output):
    """Записать ссылки на записи и отчёт только в новый каталог, без текстов."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("manifest.json", result.manifest), ("report.json", result.report)):
        with (output / name).open("x", encoding="utf-8") as target:
            target.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Подготовить групповое разделение 80/10/10 без обучения")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Новый каталог для manifest.json и report.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = prepare_dataset(args.input, args.seed)
        save_split(result, args.output)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(result.report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
