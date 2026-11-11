# AdsTextClassification

Классификатор обучается локально на размеченных объявлениях и выдаёт одну категорию для каждого сообщения. Одним комплектом модели пользуются пакетный прогноз по таблице, Flask API и классификатор Telegram Finder.

Для этого checkout в WSL выполните `cd /mnt/c/Dev/AdsTextClassification`. В Windows этот же каталог расположен по пути `C:\Dev\AdsTextClassification`. Все команды ниже выполняются из корня проекта в WSL или Linux x86-64. Нужны uv и Python 3.12. Версии зависимостей закреплены в `pyproject.toml` и `uv.lock`.

```bash
uv sync --locked --extra cpu
uv run --locked --extra cpu python train.py --input data/train_data/ads.csv
uv run --locked --extra cpu python run_predict_data.py --input data/for_prediction_data/messages.xlsx --output data/output_data/predictions.csv
uv run --locked --extra cpu python -m AppFlaskPyTorch.app
```

Для обучения на NVIDIA в WSL выполните `bash scripts/train_cuda.sh` и выберите таблицу. Путь можно передать напрямую через `--input data/train_data/ads.csv`.

Для обучения нужны столбцы `text` и `label`, для прогноза достаточно `text`. Обе команды принимают CSV и XLSX. Запуск train.py без --input открывает выбор файла из data/train_data. По умолчанию обучение записывает три файла в saved/model. Прогноз без аргументов читает data/for_prediction_data/output.xlsx. Расширение --output определяет формат результата. Если путь результата опущен, используется формат входного файла.

Разметку с несколькими категориями через `|` можно проверить без обучения: `uv run --locked --extra cpu python -m utils.training_data --input data/train_data/labeled.csv`. Подготовка исключает `review`, ошибки разметки и незавершённые проверки; [правила и справочник категорий](docs/TRAINING.md) описывают отчёт и допуск строк. Полное multilabel-обучение ещё в разработке: текущий `train.py` останавливается, если после подготовки остались строки с несколькими метками.

Для подготовки воспроизводимой выборки выполните `uv run --locked --extra cpu python -m utils.dataset_split --input data/train_data/labeled.csv --output data/prepared/candidate-split --seed 42`. Новый каталог получит манифест состава и отчёт. Группы по ID и нормализованному тексту не пересекаются между обучением, валидацией и тестом; конфликтующая разметка исключается, повторы удаляются. Текущее однокатегорийное обучение использует такое же разделение 80/10/10 и поддерживает `--seed`.

[Документация](docs/README.md) · [Контейнер и HTTP API](AppFlaskPyTorch/README.md) · [Правила работы](AGENTS.md)
