import json
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from config import config
from models.model import SimpleNN
from utils.util import get_label_dict


# Загрузка обученной модели и векторизатора
vectorizer: CountVectorizer = joblib.load(f'{config.MODEL_PATH}/vectorizer.pkl')

# Загружаем параметры модели, совпадающие с обучением
input_size = vectorizer.vocabulary_  # Получаем размер словаря из векторизатора
num_classes = 13  # Замените на количество классов, которые использовались при обучении

model = SimpleNN(input_size=len(input_size), hidden_size=100, num_classes=num_classes)
model.load_state_dict(torch.load(f'{config.MODEL_PATH}/model.pth'))
model.eval()

# Теперь предсказываем на новых данных
df: pd.DataFrame = pd.read_excel('data/test_data/output.xlsx')
X_new = df['text']

# Преобразуем новые данные с использованием загруженного векторизатора
X_new_vectors = vectorizer.transform(X_new)

# Прогоняем через модель
with torch.no_grad():
    outputs = model(torch.FloatTensor(X_new_vectors.toarray()))
    _, predicted = torch.max(outputs.data, 1)


df['predicted_label'] = predicted.numpy()
label_dict = get_label_dict()
df['label'] = df['predicted_label'].apply(lambda x: label_dict.get(x, 'Unknown'))

df.to_excel(f'{config.OUTPUT_DATA_PATH}/labeled_predicted_data.xlsx', index=False)

