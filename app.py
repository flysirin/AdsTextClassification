# # app.py
# import joblib
# import torch
# from sklearn.feature_extraction.text import CountVectorizer
#
# from config import config
# from models.model import SimpleNN
# from utils.util import get_label_dict
#
# # from flask import Flask, request, jsonify
# # Определяем архитектуру модели
# input_size = 33514
# hidden_size = 100
# num_classes = 13
#
# label_dict: dict = get_label_dict()  # Load dict for decode labels
#
#
# # input_size: int = len(vectorizer.vocabulary_)  # get size dict from vectorizer
# # num_classes: int = len(label_dict)  # get count classes (topics) then used for studied
#
# model = SimpleNN(input_size=input_size, hidden_size=100, num_classes=num_classes)
# model.load_state_dict(torch.load(f'{config.MODEL_PATH}/model.pth', weights_only=True))
# model.eval()
#
#
# data = {"inputs": ['Очки пластиковые, декоративные, 15 лари за все',
#                    'дам денег за роботу + в лc',
#                    'Он жив, просто уехал 😅']}
#
# X_new = data['inputs']
# vectorizer: CountVectorizer = joblib.load(f'{config.MODEL_PATH}/vectorizer.pkl')
# X_new_vectors = vectorizer.transform(X_new)
#
# with torch.no_grad():
#     outputs = model(torch.FloatTensor(X_new_vectors.toarray()))
#     _, predicted = torch.max(outputs.data, 1)
#
#
# res = predicted.tolist()
#
# print(res)

numbers = [1, 2, 3, 4, 5]
for number in numbers:
    if number % 2 == 0:
        print(f"{number} — четное число")
    else:
        print(f"{number} — нечетное число")
else:
    print("Все числа проверены")
