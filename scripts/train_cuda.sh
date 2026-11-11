#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'bash scripts/train_cuda.sh [--input ФАЙЛ.csv|ФАЙЛ.xlsx] [--epochs N] [--model-dir КАТАЛОГ]'
    echo 'Без --input обучение предложит выбрать таблицу в data/train_data/.'
    exit 0
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv-cuda}"
echo 'Подготовка CUDA-окружения через uv и запуск обучения…'
exec uv run --locked --extra cuda python train.py "$@" --device cuda
