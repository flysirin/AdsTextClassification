import pandas as pd
# import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import torch
import torch.nn as nn
import torch.optim as optim

from models.model import SimpleNN
from config import config
from utils.util import prepared_data_excel, get_input_data_file_excel

file_path: str = get_input_data_file_excel(config.INPUT_DATA_PATH)
df: pd.DataFrame = prepared_data_excel(file_path)


X = df['text']
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


model = SimpleNN(input_size=X_train_vectors.shape[1],
                 hidden_size=100,
                 num_classes=len(df['label'].unique()))


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 20
for epoch in range(num_epochs):
    # Прямое распространение
    outputs = model(torch.FloatTensor(X_train_vectors.toarray()))
    loss = criterion(outputs, torch.LongTensor(y_train.to_numpy()))

    # Обратное распространение и оптимизация
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


torch.save(model.state_dict(), f'{config.MODEL_PATH}/model.pth')

