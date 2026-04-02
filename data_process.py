import os
import gc
import numpy as np
import pandas as pd

def process_attribution_data(base_dir, output_csv):
    """
    遍历指定目录下的真实与预测数据，聚合生成用于归因分析的面板数据。
    """
    
    # 1. 城市经纬度字典 (基于提供的数据)
    locations = {
        "shanghai":  (31.23, 121.47),
        "chengdu":   (30.67, 104.06),
        "hohhot":    (40.84, 111.75),
        "qingdao":   (36.06, 120.38),
        "guangzhou": (23.13, 113.26),
        "urumqi":    (43.82, 87.62),
        "lhasa":     (29.65, 91.11),
        "golmud":    (36.40, 94.90),
        "kunming":   (25.04, 102.71),
        "wuhan":     (30.59, 114.30),
        "fuzhou":    (26.08, 119.30),
        "haikou":    (20.03, 110.32)
    }

    # 2. 模型列表
    models = ["cnn", "fno", "kan", "convlstm", "convgru", "trajgru", "predrnn", "unet"]
    
    # 3. 待填补的地形先验数据 (关键问题：必须填入客观的地形指标)
    # 格式: { "city": (elevation_m, roughness_index) }
    geo_features = {
        "shanghai":  (3.83, 1.0263),
        "chengdu":   (439.55, 191.3908),
        "hohhot":    (1022.23, 106.0301),
        "qingdao":   (31.89, 16.6657),
        "guangzhou": (37.24, 27.1315),
        "urumqi":    (417.98, 266.9159),
        "lhasa":     (1103.81, 642.8126),
        "golmud":    (2661.31, 60.0182),
        "kunming":   (1422.06, 114.7463),
        "wuhan":     (23.39, 5.5087),
        "fuzhou":    (93.06, 122.5838),
        "haikou":    (23.1, 18.3438)
    }
    
    all_records = []
    total_files = len(locations) * len(models)
    processed_count = 0

    print(f"初始化处理流程，预期读取 {total_files} 组数据...")

    # 4. 动态读取、计算与释放
    for city, (lat, lon) in locations.items():
        elevation, roughness = geo_features.get(city, (np.nan, np.nan))
        
        for model in models:
            # 根据提供的路径结构拼接绝对路径
            # e.g., 24-24/analysis_results_chengdu/convlstm/final_eval_point_pred.npy
            target_dir = os.path.join(base_dir, f"analysis_results_{city}", model)
            pred_path = os.path.join(target_dir, "final_eval_point_pred.npy")
            true_path = os.path.join(target_dir, "final_eval_point_true.npy")
            
            if not (os.path.exists(pred_path) and os.path.exists(true_path)):
                print(f"[-] 缺失文件，跳过: {city} - {model} (路径: {target_dir})")
                continue
                
            try:
                # 载入数据，预期维度 (34816, 24)
                pred = np.load(pred_path)
                true = np.load(true_path)
                
                # 计算均方根误差 (按时间步长)，生成 (24,) 的一维数组
                rmse_array = np.sqrt(np.mean((pred - true) ** 2, axis=0))
                
                # 确定模型分类 (架构特征)
                if model in ['convlstm', 'convgru', 'trajgru', 'predrnn']:
                    model_family = 'RNN'
                elif model in ['fno', 'kan']:
                    model_family = 'Operator'
                else:
                    model_family = 'CNN'

                # 按 24 个预报步长展开记录
                for lead_time_idx, rmse_val in enumerate(rmse_array):
                    all_records.append({
                        'City': city,
                        'Lat': lat,
                        'Lon': lon,
                        'Elevation': elevation,
                        'Roughness': roughness,
                        'Model': model,
                        'Model_Family': model_family,
                        'Lead_Time': lead_time_idx + 1,
                        'RMSE': float(rmse_val)
                    })
                    
                # 主动释放内存资源
                del pred, true, rmse_array
                gc.collect()
                
            except Exception as e:
                print(f"[!] 处理 {city}-{model} 时发生错误: {e}")
                continue
                
            processed_count += 1
            if processed_count % 16 == 0:
                print(f"[+] 进度: {processed_count} / {total_files}")

    # 5. 保存结果
    if all_records:
        df_panel = pd.DataFrame(all_records)
        df_panel.to_csv(output_csv, index=False)
        print(f"\n处理完毕。共提取 {len(df_panel)} 条归因面板记录。")
        print(f"数据已导出至: {output_csv}")
    else:
        print("\n未提取到任何有效数据，请检查基础路径是否正确。")

if __name__ == "__main__":
    # 指定基础目录为 "24-24"
    BASE_DIRECTORY = "24-24" 
    OUTPUT_FILE = "attribution_panel_data.csv"
    
    process_attribution_data(base_dir=BASE_DIRECTORY, output_csv=OUTPUT_FILE)