import json
from config import config



def get_label_dict() -> dict:
    with open(f"{config.MODEL_PATH}/labels_encoded.json") as file:
        dict_labels = json.load(file)
        dict_labels = {int(k): v for k, v in dict_labels.items()}
    return dict_labels
