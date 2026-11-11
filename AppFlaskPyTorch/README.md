# Flask API

Команды выполняются из корня AdsTextClassification. Контекст сборки включает общий uv.lock и код модели. Контейнер работает на CPU. Один процесс Gunicorn держит одну загруженную модель.

```bash
docker build -t ads-classifier .
docker run --rm -p 127.0.0.1:5000:5000 -v "$PWD/saved/model:/models:ro" ads-classifier
```

Дополнительный Dockerfile в этом каталоге также использует контекст корня: `docker build -f AppFlaskPyTorch/Dockerfile -t ads-classifier .`. Для локального запуска используйте `uv run --locked --extra cpu python -m AppFlaskPyTorch.app`. MODEL_PATH указывает каталог с model.pth, vectorizer.pkl и labels_encoded.json. Без переопределения локальный сервер использует saved/model в корне. При наличии только прежнего комплекта в AppFlaskPyTorch/saved/model он читает его оттуда.

```bash
curl -s http://127.0.0.1:5000/health
curl -s http://127.0.0.1:5000/predict -H 'Content-Type: application/json' -d '{"messages":{"42":"Продам стол"}}'
```

GET /health возвращает HTTP 200 для работающего сервиса. Поле model_ready показывает готовность к прогнозу. POST /predict принимает от 1 до 256 непустых текстов в объекте messages и возвращает predictions с исходными ключами. Пример ответа при наличии соответствующей категории — `{"predictions":{"42":"мебель"}}`. При отсутствии или несовместимости модели прогноз возвращает 503. Некорректный пакет получает 400. Для обновления модели перезапустите процесс с новым комплектом.

Обучение с `--multilabel` создаёт отдельный кандидат с прежними тремя файлами. Текущий сервер проверяет необязательный атрибут `classification_task_` в CountVectorizer и отклоняет multilabel до выбора через argmax. В этом случае `/health` возвращает `model_ready=false` и `error=model_load_failed`, а `/predict` — 503. Старые векторизаторы без атрибута считаются singlelabel и загружаются как прежде. Формат ответа со списком категорий и переход Telegram Finder относятся к следующим чекпоинтам [Roadmap](../docs/ROADMAP.md); такой кандидат пока не является релизом для рабочего сервиса.
