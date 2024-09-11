import torch
import joblib
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from config import config
from models.model import SimpleNN
from utils.util import get_label_dict


app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def predict():
    vectorizer: CountVectorizer = joblib.load(f'{config.MODEL_PATH}/vectorizer.pkl')

    label_dict: dict = get_label_dict()  # Load dict for decode labels
    input_size: int = len(vectorizer.vocabulary_)  # get size dict from vectorizer
    num_classes: int = len(label_dict)  # get count classes (topics) then used for studied

    model = SimpleNN(input_size=input_size, hidden_size=100, num_classes=num_classes)
    model.load_state_dict(torch.load(f'{config.MODEL_PATH}/model.pth', weights_only=True))
    model.eval()

    data_request = request.json

    raw_data: dict = data_request['messages']

    key_ids_sort: list[int] = sorted(raw_data)
    text_sort: list[str] = [raw_data.get(k) for k in key_ids_sort]

    X_new = text_sort
    X_new_vectors = vectorizer.transform(X_new)

    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_new_vectors.toarray()))
        _, predicted = torch.max(outputs.data, 1)

    encoded_labels: list[int] = predicted.tolist()
    decoded_labels: list[str] = [label_dict[key] for key in encoded_labels]

    result: dict[int, str] = dict(zip(key_ids_sort, decoded_labels))

    return jsonify({'predictions': result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
