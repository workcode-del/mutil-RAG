$ErrorActionPreference = "Stop"

if (-not $env:CONDA_PREFIX) {
    throw "未检测到已激活的Conda环境，请先执行 conda activate <环境名>。"
}

$activeEnvironment = if ($env:CONDA_DEFAULT_ENV) { $env:CONDA_DEFAULT_ENV } else { Split-Path -Leaf $env:CONDA_PREFIX }
if ($activeEnvironment -eq "base") {
    throw "请先激活项目专用Conda环境，不要把依赖安装到base环境。"
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

Write-Host "已为当前Conda环境 '$activeEnvironment' 配置清华Conda和PyPI镜像。"
conda config --env --show channels
python -m pip config --site list
