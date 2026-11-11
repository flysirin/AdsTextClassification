import json
import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)
ARTIFACTS = ("model.pth", "vectorizer.pkl", "labels_encoded.json")


class Predictor:
    def __init__(self, directory):
        import joblib
        import torch
        from models.model import SimpleNN

        self.vectorizer = joblib.load(directory / "vectorizer.pkl")
        if getattr(self.vectorizer, "classification_task_", "singlelabel") != "singlelabel":
            raise ValueError("Multilabel candidate requires the new prediction contract; it cannot use argmax")
        with (directory / "labels_encoded.json").open(encoding="utf-8") as file:
            self.labels = {int(key): value for key, value in json.load(file).items()}
        if not self.labels or set(self.labels) != set(range(len(self.labels))):
            raise ValueError("Label indices must be contiguous starting at zero")
        if not all(isinstance(label, str) for label in self.labels.values()):
            raise ValueError("Labels must be strings")
        self.model = SimpleNN(len(self.vectorizer.vocabulary_), 100, len(self.labels))
        self.model.load_state_dict(torch.load(directory / "model.pth", map_location="cpu", weights_only=True))
        self.model.eval()
        self.predict({"probe": "model readiness check"})

    def predict(self, messages):
        import torch

        ids = list(messages)
        vectors = self.vectorizer.transform([messages[key] for key in ids])
        with torch.inference_mode():
            indices = self.model(torch.tensor(vectors.toarray(), dtype=torch.float32)).argmax(dim=1).tolist()
        return {key: self.labels[index] for key, index in zip(ids, indices)}


def create_app(model_dir=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    legacy = Path(__file__).resolve().parent / "saved/model"
    shared = legacy.parents[2] / "saved/model"
    default = legacy if (legacy / "model.pth").is_file() and not (shared / "model.pth").is_file() else shared
    directory = Path(model_dir or os.environ.get("MODEL_PATH") or default)
    missing = [name for name in ARTIFACTS if not (directory / name).is_file()]
    predictor = None
    error = "model_missing" if missing else None
    if not missing:
        try:
            predictor = Predictor(directory)
        except Exception:
            logger.exception("Could not load classifier model from %s", directory)
            error = "model_load_failed"

    @app.get("/health")
    def health():
        return jsonify(status="running", model_ready=predictor is not None, error=error, missing=missing)

    @app.post("/predict")
    def predict():
        data = request.get_json(silent=True)
        messages = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(messages, dict) or not messages or len(messages) > 256:
            return jsonify(error="invalid_messages", message="Provide 1 to 256 ID-to-text entries in messages."), 400
        if not all(isinstance(text, str) and text.strip() for text in messages.values()):
            return jsonify(error="invalid_text", message="Each message must contain non-empty text."), 400
        if predictor is None:
            return jsonify(error=error, message="Install a complete compatible model and restart the classifier.",
                           missing=missing), 503
        try:
            return jsonify(predictions=predictor.predict(messages))
        except Exception:
            logger.exception("Prediction failed")
            return jsonify(error="prediction_failed"), 500

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
