# AdsTextClassification

После обучения multilabel-кандидата доступны [подбор порогов и независимая оценка](docs/EVALUATION.md) без переобучения и изменения файлов модели.

Классификатор обучается локально на размеченных объявлениях. Доступны однокатегорийное обучение и обучение multilabel-кандидатов. Пакетный прогноз, Flask API и Telegram Finder пока используют выдачу одной категории.

Для этого checkout в WSL выполните `cd /mnt/c/Dev/AdsTextClassification`. В Windows этот же каталог расположен по пути `C:\Dev\AdsTextClassification`. Все команды ниже выполняются из корня проекта в WSL или Linux x86-64. Нужны uv и Python 3.12. Версии зависимостей закреплены в `pyproject.toml` и `uv.lock`.

```bash
uv sync --locked --extra cpu
uv run --locked --extra cpu python train.py --input data/train_data/ads.csv
uv run --locked --extra cpu python run_predict_data.py --input data/for_prediction_data/messages.xlsx --output data/output_data/predictions.csv
uv run --locked --extra cpu python -m AppFlaskPyTorch.app
```

Для обучения на NVIDIA в WSL выполните `bash scripts/train_cuda.sh`. Без аргументов откроется мастер: режим multilabel или singlelabel → таблица → число эпох → каталог модели. Enter принимает предложенное значение. Каталоги берутся из `config/config.py`: таблицы из `TRAIN_DATA_PATH`, новый каталог кандидата рядом с `MODEL_PATH`. По умолчанию выбраны multilabel, `labeled.csv` при наличии, 100 эпох и имя модели с датой. Имя и остальные значения можно изменить. Запуск с аргументами, например `--input data/train_data/ads.csv`, сохраняет прежний командный режим.

Для обучения нужны столбцы `text` и `label`, для прогноза достаточно `text`. Обе команды принимают CSV и XLSX. Запуск train.py без --input открывает выбор файла из data/train_data. По умолчанию обучение записывает три файла в saved/model. Прогноз без аргументов читает data/for_prediction_data/output.xlsx. Расширение --output определяет формат результата. Если путь результата опущен, используется формат входного файла.

Разметку с несколькими категориями через `|` можно проверить без обучения: `uv run --locked --extra cpu python -m utils.training_data --input data/train_data/labeled.csv`. Подготовка исключает `review`, ошибки разметки и незавершённые проверки; [правила и справочник категорий](docs/TRAINING.md) описывают отчёт и допуск строк.

Для обучения multilabel-кандидата выполните `uv run --locked --extra cpu python train.py --multilabel --input data/train_data/labeled.csv --epochs 100 --model-dir saved/candidate-multilabel`. На NVIDIA используйте `bash scripts/train_cuda.sh` с теми же аргументами. Каталог кандидата должен быть новым и находиться вне `saved/model/`. Доступны `--batch-size`, `--seed` и необязательный `--pos-weight`. Модель обучается с BCEWithLogitsLoss; текущие загрузчики прогноза отклоняют этот режим до реализации порогов и нового API по [Roadmap](docs/ROADMAP.md).

Для подготовки воспроизводимой выборки выполните `uv run --locked --extra cpu python -m utils.dataset_split --input data/train_data/labeled.csv --output data/prepared/candidate-split --seed 42`. Новый каталог получит манифест состава и отчёт. Группы по ID и нормализованному тексту не пересекаются между обучением, валидацией и тестом; конфликтующая разметка исключается, повторы удаляются. Текущее однокатегорийное обучение использует такое же разделение 80/10/10 и поддерживает `--seed`.

[Документация](docs/README.md) · [Контейнер и HTTP API](AppFlaskPyTorch/README.md) · [Правила работы](AGENTS.md)
## Способы обучения

В мастере `bash scripts/train_cuda.sh` для multilabel можно выбрать новый способ с лучшей эпохой и ранней остановкой либо старый со всеми эпохами и последними весами. Для команд с аргументами новый способ включается через `--training-mode best`, старый остаётся по умолчанию (`legacy`). Настройки: `--patience 5 --min-delta 0.0001`; `--no-early-stopping` отключает остановку, сохраняя выбор лучших весов. Подробнее: [обучение](docs/TRAINING.md).

