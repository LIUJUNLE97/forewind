#!/bin/bash

PROJECT_ACCOUNT="${WIND_PROJECT_ACCOUNT:-YOUR_PROJECT_ACCOUNT}"
PROJECT_DIR=${WIND_PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}

# 虚拟环境路径
VENV_PATH="${WIND_VENV_PATH:-${PROJECT_DIR}/.venv}"

# 数据路径
NPY_DATA_DIR="${WIND_NPY_DATA_DIR:-${PROJECT_DIR}/npy_data}"
SOM_MAP_PATH="${WIND_SOM_MAP:-${PROJECT_DIR}/data/metadata/som_map.npy}"


# 2. 实验超参数配置

EPOCHS=100               
SEQ_LEN=168               
PRED_LEN=24              
PATIENCE=10              

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
    "fno" 
    "kan" 
    "convlstm"
    "convgru"
    "trajgru"
    "predrnn"
    "unet"
)

declare -A MODEL_BATCH_SIZE=(
    #["cnn"]=2048   
    #["fno"]=1024   
    #["kan"]=128        
    #["convlstm"]=128
    #["convgru"]=128   
    #["trajgru"]=64      
    #["predrnn"]=64    
    #["unet"]=1024
    
    ["cnn"]=512
    ["fno"]=512
    ["kan"]=32 
    ["convlstm"]=32
    ["convgru"]=32
    ["trajgru"]=16 
    ["predrnn"]=16  
    ["unet"]=512


    
)

declare -A MODEL_TIME_LIMIT=(
    ["cnn"]="2-00:00:00"      
    ["fno"]="2-00:00:00"      
    ["kan"]="2-00:00:00"      
    ["convlstm"]="4-00:00:00" 
    ["convgru"]="4-00:00:00"  
    ["trajgru"]="6-00:00:00"  
    ["predrnn"]="6-00:00:00"  
    ["unet"]="2-00:00:00"     
)

echo "Regions to process:"
for region in "${!LOCATIONS[@]}"; do
    coords="${LOCATIONS[$region]}"
    lat=$(echo $coords | cut -d' ' -f1)
    lon=$(echo $coords | cut -d' ' -f2)
    printf "  %-15s -> (%.2f, %.2f)\n" "$region" "$lat" "$lon"
done
echo ""
echo "Models per region:"
for model in "${models[@]}"; do
    batch_size=${MODEL_BATCH_SIZE[$model]}
    time_limit=${MODEL_TIME_LIMIT[$model]}
    printf "  %-12s -> Batch: %-6s  Time: %s\n" "$model" "$batch_size" "$time_limit"
done
echo ""
echo "Total jobs: ${#LOCATIONS[@]} regions × ${#models[@]} models = $((${#LOCATIONS[@]} * ${#models[@]})) jobs"
echo ""
echo "------------------------------------------"

# 统计变量
total_submitted=0
total_failed=0

for region in "${!LOCATIONS[@]}"; do
    
    echo ""
    echo "=========================================="
    echo "REGION: $region"
    echo "=========================================="
    
    # 解析坐标
    coords="${LOCATIONS[$region]}"
    lat=$(echo $coords | cut -d' ' -f1)
    lon=$(echo $coords | cut -d' ' -f2)
    
    LOG_DIR="${PROJECT_DIR}/7-24/logs_slurm_${region}"
    OUTPUT_DIR="${PROJECT_DIR}/7-24/analysis_results_${region}"
    
    # 创建目录
    mkdir -p "$LOG_DIR"
    mkdir -p "$OUTPUT_DIR"
    
    echo "  Coordinates: ($lat, $lon)"
    echo "  Log Dir: $LOG_DIR"
    echo "  Output Dir: $OUTPUT_DIR"
    echo ""
    
    # 遍历模型
    for model in "${models[@]}"; do
        
        # 获取模型特定配置
        batch_size=${MODEL_BATCH_SIZE[$model]}
        time_limit=${MODEL_TIME_LIMIT[$model]:-"2-00:00:00"}
        
        # 检查batch size是否定义
        if [ -z "$batch_size" ]; then
            echo "  ⚠️  Warning: Batch size not defined for '$model', using default 128"
            batch_size=128
        fi

        job_name="wind_${model}"
        job_script_path="${LOG_DIR}/${job_name}.sh"
        
        echo "  => Generating: $model (Batch: $batch_size)"
        
        cat > "$job_script_path" << EOF
#!/bin/bash
#SBATCH -A ${PROJECT_ACCOUNT}
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100fat:1
#SBATCH -t ${time_limit}
#SBATCH --cpus-per-task=8
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err

# ==========================================
# 运行时环境设置
# ==========================================
echo "=========================================="
echo "Job Info:"
echo "  Region: ${region} (${lat}, ${lon})"
echo "  Model: ${model}"
echo "  Batch Size: ${batch_size}"
echo "  GPU: \${SLURM_JOB_GPUS}"
echo "  Node: \$(hostname)"
echo "  Start Time: \$(date)"
echo "=========================================="
echo ""

# 1. 激活环境
source ${VENV_PATH}/bin/activate

# 2. 设置 Python 路径
export PYTHONPATH="\${PYTHONPATH}:${PROJECT_DIR}"

# 3. 检查环境
echo "Python Version:"
python --version
echo ""

echo "PyTorch & CUDA Info:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}')"
echo ""

echo "GPU Info:"
nvidia-smi
echo ""

# ==========================================
# 运行训练
# ==========================================
cd ${PROJECT_DIR}

echo "Starting training for ${region} - ${model}..."
echo ""

python -u src_2/run_models.py \\
    --model_name ${model} \\
    --nc_dir ${NPY_DATA_DIR} \\
    --som_map_path ${SOM_MAP_PATH} \\
    --target_lat ${lat} \\
    --target_lon ${lon} \\
    --epochs ${EPOCHS} \\
    --batch_size ${batch_size} \\
    --seq_len ${SEQ_LEN} \\
    --pred_len ${PRED_LEN} \\
    --patience ${PATIENCE} \\
    --lr 0.001 \\
    --num_workers 4 \\
    --output_dir ${OUTPUT_DIR}

EXIT_CODE=\$?

# ==========================================
# 任务结束
# ==========================================
echo ""
echo "=========================================="
echo "Job finished at \$(date)"
echo "Exit code: \${EXIT_CODE}"
echo "=========================================="

exit \${EXIT_CODE}
EOF

        # 给脚本添加执行权限
        chmod +x "$job_script_path"
        
       
        echo "     Submitting to SLURM..."
        sbatch_output=$(sbatch "$job_script_path" 2>&1)
        
        if [ $? -eq 0 ]; then
            job_id=$(echo "$sbatch_output" | grep -oP '(?<=Submitted batch job )\d+')
            echo "     Submitted successfully (Job ID: $job_id)"
            ((total_submitted++))
        else
            echo "     Submission failed: $sbatch_output"
            ((total_failed++))
        fi
        
        # 延迟提交，避免过载调度器
        sleep 1
    done
    
    echo ""
    echo "  Completed $region: ${#models[@]} jobs submitted"
    echo ""
done

echo ""
echo "=========================================="
echo "Batch Submission Complete!"
echo ""
echo "Summary:"
echo "  Total jobs submitted: $total_submitted"
echo "  Total jobs failed:    $total_failed"
echo "  Expected total:       $((${#LOCATIONS[@]} * ${#models[@]}))"
echo ""
echo "Monitoring commands:"
echo "  - Check queue:        squeue -u \$USER"
echo "  - Check all logs:     ls -lt logs_slurm_*/"
echo "  - Watch specific log: tail -f logs_slurm_shanghai/wind_cnn_*.out"
echo "  - Cancel all:         scancel -u \$USER"
echo ""
echo "Output directories:"
for region in "${!LOCATIONS[@]}"; do
    echo "  - $region: ${PROJECT_DIR}/analysis_results_${region}"
done
echo ""
echo "=========================================="
