import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from config.config import INPUT_DATA_PATH


def prepared_data_excel(file_path: str, label_name: str = 'label') -> pd.DataFrame:
    df = pd.read_excel(file_path)
    labeled_df = df.dropna(subset=[f'{label_name}'])  # Keep only labeled data

    label_encoder = LabelEncoder()
    labeled_df['label_encoded'] = label_encoder.fit_transform(labeled_df['label'])

    return labeled_df


def get_input_data_file_excel(dir_path: str = INPUT_DATA_PATH,
                              label_name: str = 'label') -> None | str:
    files: list = list(Path(dir_path).glob('*.xlsx'))
    i = 0
    for i, file in enumerate(files, 1):
        print(f"Choose file:")
        print(f"filename: {file} / number - {i}")
    if i > 0:
        x = input("number: ")
        if x.isdigit() and int(x) <= i:
            return files[int(x)]
    else:
        raise ValueError(f"No file found")
