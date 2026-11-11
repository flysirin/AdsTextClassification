#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'bash scripts/train_cuda.sh [--input ФАЙЛ.csv|ФАЙЛ.xlsx] [--epochs N] [--model-dir КАТАЛОГ] [--multilabel] [--training-mode legacy|best] [--patience N] [--min-delta X] [--no-early-stopping] [--pos-weight] [--batch-size N] [--seed N]'
    echo 'Без аргументов запускается выбор режима, таблицы, эпох и каталога модели из настроек config/config.py.'
    exit 0
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv-cuda}"
echo 'Подготовка CUDA-окружения через uv и запуск обучения…'
if [[ $# -eq 0 ]]; then
    exec uv run --locked --extra cuda python -m scripts.train_interactive --device cuda
fi
exec uv run --locked --extra cuda python train.py "$@" --device cuda
