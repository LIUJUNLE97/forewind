#!/bin/bash
#SBATCH -A YOUR_PROJECT_ACCOUNT   # 替换为本地集群的项目账号
#SBATCH -p alvis                     # 申请 alvis 队列
#SBATCH -N 1                         # 1 个节点
#SBATCH --gpus-per-node=A40:1            # 申请 1 块 GPU（如果是纯 CPU 评估可不加）
#SBATCH -t 00:20:00                  # 限制时间 20 分钟
#SBATCH -J wind_eval                 # 任务名称

# 激活可选虚拟环境；也可以预先激活环境后直接运行。
PROJECT_ROOT=${WIND_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
if [ -n "${WIND_VENV_PATH:-}" ] && [ -f "${WIND_VENV_PATH}/bin/activate" ]; then
    source "${WIND_VENV_PATH}/bin/activate"
fi
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src_2:${PYTHONPATH:-}"

# 运行评估
python "${PROJECT_ROOT}/src_2/eval.py" --model_name cnn
