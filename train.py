import json

import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import torch
import torch.nn as nn
import torch.optim as optim

from models.model import SimpleNN
from config import config
from utils.util import prepared_data_excel, get_input_data_file_excel

from datetime import datetime

file_path: str = get_input_data_file_excel(config.TRAIN_DATA_PATH)
df: pd.DataFrame = prepared_data_excel(file_path)  # prepared data

X = df['text']
y = df['label_encoded']

# Data Splitting. Will split the data into training and test sets to check how well the model performs
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Convert text to numeric data. One popular method is to use a Bag of Words or TF-IDF
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

torch.set_num_threads(4)
print(torch.get_num_threads())  # This should print 5
# Create a simple neural network in PyTorch to classify messages
model = SimpleNN(input_size=X_train_vectors.shape[1],
                 hidden_size=100,
                 num_classes=len(df['label'].unique()))

# Optimization and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Model training
num_epochs = 100
for epoch in range(num_epochs):
    # Direct distribution
    outputs = model(torch.FloatTensor(X_train_vectors.toarray()))
    loss = criterion(outputs, torch.LongTensor(y_train.to_numpy()))

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

date_time = datetime.now().strftime("%Y%m%d_%H-%M")

# After training the model evaluate it on the test set
with torch.no_grad():
    outputs = model(torch.FloatTensor(X_test_vectors.toarray()))
    _, predicted = torch.max(outputs.data, 1)

    # # Accuracy assessment
    # accuracy = (predicted == torch.LongTensor(y_test.to_numpy())).sum().item() / len(y_test)
    # print(f'Accuracy: {accuracy * 100:.2f}%')

y_test_tensor = torch.LongTensor(y_test.to_numpy())

# Calculating metrics
precision = precision_score(y_test_tensor, predicted, average='weighted')  # 'weighted' для многоклассовой классификации
recall = recall_score(y_test_tensor, predicted, average='weighted')
f1 = f1_score(y_test_tensor, predicted, average='weighted')
accuracy = accuracy_score(y_test_tensor, predicted)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1-score: {f1 * 100:.2f}%')

# Save model and settings
torch.save(model.state_dict(), f'{config.MODEL_PATH}/model.pth')
joblib.dump(vectorizer, f'{config.MODEL_PATH}/vectorizer.pkl')

labels_dict: dict = dict(zip(df['label_encoded'], df['label']))
with open(f'{config.MODEL_PATH}/labels_encoded.json', mode="w", encoding="utf-8") as file:
    json.dump(labels_dict, file, default=int)
