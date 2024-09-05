import json

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

from config import config


# from config.config import INPUT_DATA_PATH


def prepared_data_excel(file_path: str, label_name: str = 'label') -> pd.DataFrame:
    """Convert 'label' to number 'label_encoded' """

    df = pd.read_excel(file_path)
    labeled_df = df.dropna(subset=[f'{label_name}'])  # Keep only labeled data

    label_encoder = LabelEncoder()
    labeled_df['label_encoded'] = label_encoder.fit_transform(labeled_df['label'])

    return labeled_df


def get_input_data_file_excel(dir_path: str = config.TRAIN_DATA_PATH,
                              label_name: str = 'label') -> None | str:
    files: list = list(Path(dir_path).glob('*.xlsx'))
    i = 0
    for i, file in enumerate(files, 1):
        print(f"Choose file:")
        print(f"filename: {file} / number - {i}")
    if i > 0:
        x = input("Input number?: ")
        if x.isdigit() and int(x) <= i:
            return files[int(x) - 1]
    else:
        raise FileNotFoundError(f"No file found")


def get_label_dict() -> dict:
    with open(f"{config.MODEL_PATH}/labels_encoded.json") as file:
        dict_labels = json.load(file)
        dict_labels = {int(k): v for k, v in dict_labels.items()}
    return dict_labels
