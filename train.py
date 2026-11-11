import argparse
import json
from pathlib import Path

import joblib
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

from config import config
from models.model import SimpleNN
from utils.dataset_split import split_training_data
from utils.util import get_input_data_file_excel, prepared_data_excel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path(config.MODEL_PATH))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs должен быть больше нуля")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA недоступна для текущего окружения или драйвера")
    try:
        frame = prepared_data_excel(args.input or get_input_data_file_excel())
    except ValueError as error:
        parser.error(str(error))
    if frame["labels"].map(len).gt(1).any():
        parser.error("Данные содержат несколько категорий в строке. Подготовка multilabel готова, "
                     "но обучение с BCEWithLogitsLoss ещё не реализовано (чекпоинт 3 Roadmap). "
                     "Обучение остановлено, чтобы не создавать составные классы.")
    try:
        split = split_training_data(frame, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))
    print("Разделение данных: " + json.dumps(split.report, ensure_ascii=False))
    train, test = split.partitions["train"], split.partitions["test"]
    # Валидация сохранена в split.partitions для следующего этапа подбора порогов.
    encoder = LabelEncoder().fit(train["label"])
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(train["text"])
    x_test = vectorizer.transform(test["text"])
    labels = torch.tensor(encoder.transform(train["label"]), dtype=torch.long)
    names = dict(enumerate(encoder.classes_.tolist()))
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model = SimpleNN(x.shape[1], 100, len(names)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(args.epochs):
        total = 0.0
        for ids in torch.randperm(x.shape[0]).split(128):
            batch = torch.tensor(x[ids.numpy()].toarray(), dtype=torch.float32, device=args.device)
            loss = torch.nn.functional.cross_entropy(model(batch), labels[ids].to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        print(f"Epoch {epoch + 1}/{args.epochs}, loss={total / x.shape[0]:.4f}")
    model.eval()
    predictions = []
    with torch.inference_mode():
        for offset in range(0, x_test.shape[0], 128):
            batch = torch.tensor(x_test[offset:offset+128].toarray(), dtype=torch.float32, device=args.device)
            predictions.extend(model(batch).argmax(dim=1).tolist())
    y = encoder.transform(test["label"])
    p, r, f, _ = precision_recall_fscore_support(y, predictions, average="weighted", zero_division=0)
    print(f"Accuracy={accuracy_score(y, predictions):.4f} Precision={p:.4f} Recall={r:.4f} F1={f:.4f}")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), args.model_dir / "model.pth")
    joblib.dump(vectorizer, args.model_dir / "vectorizer.pkl")
    (args.model_dir / "labels_encoded.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
