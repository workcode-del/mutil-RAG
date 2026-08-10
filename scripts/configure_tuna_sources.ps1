$ErrorActionPreference = "Stop"

if (-not $env:CONDA_PREFIX) {
    throw "请先执行 conda activate paper-rag，再配置当前环境的下载源。"
}

$activeEnvironment = Split-Path -Leaf $env:CONDA_PREFIX
if ($activeEnvironment -ne "paper-rag") {
    throw "当前激活的是 '$activeEnvironment'，请先执行 conda activate paper-rag。"
}

# Conda配置写入当前激活环境，不修改用户级或全局.condarc。
conda config --env --set channel_priority strict
conda config --env --set show_channel_urls true
conda config --env --prepend channels `
    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"

# --site只写入当前Conda环境，不修改用户级或全局pip配置。
python -m pip config --site set global.index-url `
    "https://pypi.tuna.tsinghua.edu.cn/simple"
python -m pip config --site set global.trusted-host `
    "pypi.tuna.tsinghua.edu.cn"

Write-Host "已为当前 paper-rag Conda环境配置清华Conda和PyPI镜像。"
conda config --env --show channels
python -m pip config --site list
