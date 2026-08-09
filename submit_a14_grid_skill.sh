#!/bin/bash

PROJECT_ACCOUNT="${WIND_PROJECT_ACCOUNT:-YOUR_PROJECT_ACCOUNT}"
PROJECT_DIR=${WIND_PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}

# ========= 环境路径 =========
VENV_PATH="${WIND_VENV_PATH:-${PROJECT_DIR}/.venv}"

# ========= 数据路径 =========
NPY_DATA_DIR="${WIND_NPY_DATA_DIR:-${PROJECT_DIR}/npy_data}"
SOM_MAP_PATH="${WIND_SOM_MAP:-${PROJECT_DIR}/data/metadata/som_map.npy}"

# ========= A14 输入/输出路径 =========
# 已训练好的 24->24 checkpoint 所在目录
BASE_DIR="${WIND_RESULTS_DIR:-${PROJECT_DIR}/24-24}"

# A14 输出目录：每个 city/model 会保存 64x64 ss_map / mse_map / summary
A14_OUTPUT_ROOT="${WIND_A14_OUTPUT_ROOT:-${PROJECT_DIR}/fig/a14_grid_skill}"

# ========= 实验配置 =========
SEQ_LEN=24
PRED_LEN=24
PATCH_SIZE=64
NUM_WORKERS=4

# 如果结果已经存在，默认跳过。设为 1 则强制重跑覆盖。
FORCE=0

declare -A LOCATIONS=(
    ["shanghai"]="31.23 121.47"
    ["chengdu"]="30.67 104.06"
    ["hohhot"]="40.84 111.75"
    ["qingdao"]="36.06 120.38"
    ["guangzhou"]="23.13 113.26"
    ["urumqi"]="43.82 87.62"
    ["lhasa"]="29.65 91.11"
    ["golmud"]="36.40 94.90"
    ["kunming"]="25.04 102.71"
    ["wuhan"]="30.59 114.30"
    ["fuzhou"]="26.08 119.30"
    ["haikou"]="20.03 110.32"
)

models=(
    "cnn"
   # "fno"
    #"kan"
    #"convlstm"
   # "convgru"
   # "trajgru"
   # "predrnn"
   # "unet"
)

# A14 是 inference-only，但 recurrent 模型仍然慢，batch size 保守一些
declare -A MODEL_BATCH_SIZE=(
    ["cnn"]=512
   # ["fno"]=512
   # ["kan"]=32
   # ["convlstm"]=32
    #["convgru"]=32
   # ["trajgru"]=16
   # ["predrnn"]=16
   # ["unet"]=512
)

# A14 不训练，时间可以比训练短；Golmud/Lhasa/recurrent 仍建议留余量
declare -A MODEL_TIME_LIMIT=(
    ["cnn"]="01:30:00"
    ["fno"]="01:30:00"
    ["kan"]="02:00:00"
    ["convlstm"]="03:00:00"
    ["convgru"]="03:00:00"
    ["trajgru"]="04:00:00"
    ["predrnn"]="04:00:00"
    ["unet"]="01:30:00"
)

echo "=========================================="
echo "A14 Grid-level Persistence Skill Submission"
echo "=========================================="
echo "Project dir:       ${PROJECT_DIR}"
echo "Base checkpoint:   ${BASE_DIR}"
echo "Output root:       ${A14_OUTPUT_ROOT}"
echo "Seq/Pred:          ${SEQ_LEN} -> ${PRED_LEN}"
echo "Force rerun:       ${FORCE}"
echo ""
echo "Total jobs: ${#LOCATIONS[@]} regions × ${#models[@]} models = $((${#LOCATIONS[@]} * ${#models[@]}))"
echo "=========================================="
echo ""

total_submitted=0
total_skipped=0
total_failed=0

for region in "${!LOCATIONS[@]}"; do

    echo ""
    echo "=========================================="
    echo "REGION: ${region}"
    echo "=========================================="

    coords="${LOCATIONS[$region]}"
    lat=$(echo $coords | cut -d' ' -f1)
    lon=$(echo $coords | cut -d' ' -f2)

    LOG_DIR="${PROJECT_DIR}/logs_a14_${region}"
    mkdir -p "${LOG_DIR}"

    echo "Coordinates: (${lat}, ${lon})"
    echo "Log dir: ${LOG_DIR}"
    echo ""

    for model in "${models[@]}"; do

        batch_size=${MODEL_BATCH_SIZE[$model]}
        time_limit=${MODEL_TIME_LIMIT[$model]:-"02:00:00"}

        checkpoint_path="${BASE_DIR}/analysis_results_${region}/${model}/checkpoint.pth"
        output_map="${A14_OUTPUT_ROOT}/analysis_results_${region}/${model}/ss_persist_map.npy"

        if [ ! -f "${checkpoint_path}" ]; then
            echo "  [SKIP] ${region}-${model}: missing checkpoint:"
            echo "         ${checkpoint_path}"
            ((total_skipped++))
            continue
        fi

        if [ "${FORCE}" -eq 0 ] && [ -f "${output_map}" ]; then
            echo "  [SKIP] ${region}-${model}: A14 output already exists."
            echo "         ${output_map}"
            ((total_skipped++))
            continue
        fi

        job_name="a14_${region}_${model}"
        job_script_path="${LOG_DIR}/${job_name}.sh"

        echo "  => Generating job: ${region}-${model} | Batch=${batch_size} | Time=${time_limit}"

        cat > "${job_script_path}" << EOF
#!/bin/bash
#SBATCH -A ${PROJECT_ACCOUNT}
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100fat:1
#SBATCH -t ${time_limit}
#SBATCH --cpus-per-task=8
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err

echo "=========================================="
echo "A14 Job Info"
echo "  Region:      ${region}"
echo "  Coordinates: ${lat}, ${lon}"
echo "  Model:       ${model}"
echo "  Batch size:  ${batch_size}"
echo "  Base dir:    ${BASE_DIR}"
echo "  Output root: ${A14_OUTPUT_ROOT}"
echo "  GPU:         \${SLURM_JOB_GPUS}"
echo "  Node:        \$(hostname)"
echo "  Start time:  \$(date)"
echo "=========================================="
echo ""

source ${VENV_PATH}/bin/activate

export PYTHONPATH="\${PYTHONPATH}:${PROJECT_DIR}:${PROJECT_DIR}/src_2"

cd ${PROJECT_DIR}

echo "Python:"
python --version
echo ""

echo "PyTorch / CUDA:"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU count:', torch.cuda.device_count())"
echo ""

nvidia-smi
echo ""

echo "Starting A14 grid skill computation..."
echo ""

python -u src_2/a14_grid_skill_maps.py \\
    --base_dir ${BASE_DIR} \\
    --output_root ${A14_OUTPUT_ROOT} \\
    --nc_dir ${NPY_DATA_DIR} \\
    --som_map_path ${SOM_MAP_PATH} \\
    --cities ${region} \\
    --models ${model} \\
    --seq_len ${SEQ_LEN} \\
    --pred_len ${PRED_LEN} \\
    --patch_size ${PATCH_SIZE} \\
    --batch_size ${batch_size} \\
    --num_workers ${NUM_WORKERS}

EXIT_CODE=\$?

echo ""
echo "=========================================="
echo "A14 job finished"
echo "  Region: ${region}"
echo "  Model:  ${model}"
echo "  End:    \$(date)"
echo "  Exit:   \${EXIT_CODE}"
echo "=========================================="

exit \${EXIT_CODE}
EOF

        chmod +x "${job_script_path}"

        sbatch_output=$(sbatch "${job_script_path}" 2>&1)

        if [ $? -eq 0 ]; then
            job_id=$(echo "${sbatch_output}" | grep -oP '(?<=Submitted batch job )\d+')
            echo "     Submitted successfully: Job ID ${job_id}"
            ((total_submitted++))
        else
            echo "     Submission failed: ${sbatch_output}"
            ((total_failed++))
        fi

        sleep 1
    done

done

echo ""
echo "=========================================="
echo "A14 Submission Complete"
echo "=========================================="
echo "Submitted: ${total_submitted}"
echo "Skipped:   ${total_skipped}"
echo "Failed:    ${total_failed}"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  ls -lt logs_a14_*/"
echo "  tail -f logs_a14_shanghai/a14_shanghai_cnn_*.out"
echo ""
echo "Output:"
echo "  ${A14_OUTPUT_ROOT}/analysis_results_<city>/<model>/ss_persist_map.npy"
echo "  ${A14_OUTPUT_ROOT}/analysis_results_<city>/<model>/a14_grid_skill_summary.csv"
echo "  ${A14_OUTPUT_ROOT}/a14_all_grid_skill_summary.csv"
echo "=========================================="
