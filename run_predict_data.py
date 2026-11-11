import argparse
from pathlib import Path
import joblib
from utils.tables import read_table, write_table
import torch
import json

from config import config
from models.model import SimpleNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(config.FOR_PREDICTION_PATH) / "output.xlsx")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=Path(config.MODEL_PATH))
    args = parser.parse_args()
    vectorizer = joblib.load(args.model_dir / "vectorizer.pkl")
    if getattr(vectorizer, "classification_task_", "singlelabel") != "singlelabel":
        parser.error("Multilabel-кандидат не поддерживает прежний прогноз через argmax; "
                     "новый формат прогноза запланирован в чекпоинте 5 Roadmap")
    labels = json.loads((args.model_dir / "labels_encoded.json").read_text(encoding="utf-8"))
    model = SimpleNN(len(vectorizer.vocabulary_), 100, len(labels))
    model.load_state_dict(torch.load(args.model_dir / "model.pth", map_location="cpu", weights_only=True))
    model.eval()
    frame = read_table(args.input)
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(frame), 128):
            values = frame["text"].iloc[offset:offset+128].fillna("").astype(str)
            vectors = torch.tensor(vectorizer.transform(values).toarray(), dtype=torch.float32)
            predictions.extend(model(vectors).argmax(dim=1).tolist())
    frame["predicted_label"] = predictions
    frame["label"] = [labels[str(i)] for i in predictions]
    output = args.output or Path(config.OUTPUT_DATA_PATH) / ("labeled_predicted_data" + args.input.suffix)
    write_table(frame, output)


if __name__ == "__main__":
    main()
