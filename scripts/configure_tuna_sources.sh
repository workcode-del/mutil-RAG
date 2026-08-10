#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "paper-rag" ]]; then
  echo "请先执行 conda activate paper-rag。" >&2
  exit 1
fi

# All configuration is scoped to the active Conda environment.
conda config --env --set channel_priority strict
conda config --env --set show_channel_urls true
conda config --env --prepend channels \
  "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"

python -m pip config --site set global.index-url \
  "https://pypi.tuna.tsinghua.edu.cn/simple"
python -m pip config --site set global.trusted-host \
  "pypi.tuna.tsinghua.edu.cn"

echo "已为paper-rag环境配置清华Conda和PyPI镜像。"
conda config --env --show channels
python -m pip config --site list
