import csv
from pathlib import Path

import pandas as pd


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Поддерживаются .xlsx и .csv: {path}")
    with path.open(encoding="utf-8-sig", newline="") as source:
        for sep in (",", ";"):
            source.seek(0)
            columns = next(csv.reader(source, delimiter=sep), [])
            if "text" in columns and len(columns) > 1:
                break
        else:
            # The one-column format has no field separator in its header.
            # Keep punctuation inside the message, including unquoted commas.
            sep = "\x1f"
    return pd.read_csv(path, encoding="utf-8-sig", sep=sep, dtype=str, keep_default_na=False)


def write_table(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".xlsx":
        frame.to_excel(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
    else:
        raise ValueError(f"Поддерживаются .xlsx и .csv: {path}")
