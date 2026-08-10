#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "未检测到已激活的Conda环境，请先执行 conda activate <环境名>。" >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
  echo "请先激活项目专用Conda环境，不要把依赖安装到base环境。" >&2
  exit 1
fi

active_environment="${CONDA_DEFAULT_ENV:-$(basename "$CONDA_PREFIX")}"

# All configuration is scoped to the active Conda environment.
conda config --env --set channel_priority strict
conda config --env --set show_channel_urls true
conda config --env --prepend channels \
  "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"

python -m pip config --site set global.index-url \
  "https://pypi.tuna.tsinghua.edu.cn/simple"
python -m pip config --site set global.trusted-host \
  "pypi.tuna.tsinghua.edu.cn"

echo "已为当前Conda环境 '${active_environment}' 配置清华Conda和PyPI镜像。"
conda config --env --show channels
python -m pip config --site list
