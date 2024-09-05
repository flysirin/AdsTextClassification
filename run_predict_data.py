import pandas as pd
import torch
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from config import config
from models.model import SimpleNN
from utils.util import get_label_dict

# Load vectorizer for studied model
vectorizer: CountVectorizer = joblib.load(f'{config.MODEL_PATH}/vectorizer.pkl')

label_dict: dict = get_label_dict()  # Load dict for decode labels


input_size: int = len(vectorizer.vocabulary_)  # get size dict from vectorizer
num_classes: int = len(label_dict)  # get count classes (topics) then used for studied

model = SimpleNN(input_size=input_size, hidden_size=100, num_classes=num_classes)
model.load_state_dict(torch.load(f'{config.MODEL_PATH}/model.pth', weights_only=True))
model.eval()

# Load new data for predict
df: pd.DataFrame = pd.read_excel(f'{config.FOR_PREDICTION_PATH}/output.xlsx')
X_new = df['text']

# Converting new data using the loaded vectorizer
X_new_vectors = vectorizer.transform(X_new)

# Running new data through the model
with torch.no_grad():
    outputs = model(torch.FloatTensor(X_new_vectors.toarray()))
    _, predicted = torch.max(outputs.data, 1)

df['predicted_label'] = predicted.numpy()
df['label'] = df['predicted_label'].apply(lambda x: label_dict.get(x, 'Unknown'))

df.to_excel(f'{config.OUTPUT_DATA_PATH}/labeled_predicted_data.xlsx', index=False)
