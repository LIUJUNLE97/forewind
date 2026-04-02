import pandas as pd
import os

def aggregate_results():
    # 1. 定义实验参数
    base_dir = "24-24"
    cities = [
        "shanghai", "chengdu", "hohhot", "qingdao", "guangzhou", 
        "urumqi", "lhasa", "golmud", "kunming", "wuhan", "haikou", "fuzhou"
    ]
    models = [
        "cnn", "fno", "kan", "convlstm", "convgru", "trajgru", "predrnn", "unet"
    ]
    
    all_rows = []
    missing_files = []

    print(f"开始从 {base_dir} 提取数据...")

    # 2. 遍历路径提取数据
    for city in cities:
        for model in models:
            # 构建完整的文件路径
            file_path = os.path.join(base_dir, f"analysis_results_{city}", model, "final_eval_metrics.csv")
            
            if os.path.exists(file_path):
                try:
                    # 读取 CSV
                    df = pd.read_csv(file_path)
                    
                    # 如果 CSV 为空则跳过
                    if df.empty:
                        continue
                        
                    # 添加城市和模型标签（放在表格最前面）
                    df.insert(0, 'City', city)
                    df.insert(1, 'Model', model)
                    
                    all_rows.append(df)
                except Exception as e:
                    print(f"读取错误 [{city} - {model}]: {e}")
            else:
                missing_files.append(f"{city}/{model}")

    # 3. 合并并保存
    if all_rows:
        master_df = pd.concat(all_rows, ignore_index=True)
        
        # 保存为母表
        output_name = "master_results_24-24.csv"
        master_df.to_csv(output_name, index=False)
        
        print("\n" + "="*30)
        print(f"✅ 汇总完成！")
        print(f"总计读取成功: {len(all_rows)} / {len(cities)*len(models)} 组数据")
        print(f"结果已保存至: {output_file_path if 'output_file_path' in locals() else output_name}")
        
        if missing_files:
            print(f"\n⚠️  缺失 {len(missing_files)} 组数据（可能正在运行）:")
            # 每行打印 4 个，避免太长
            for i in range(0, len(missing_files), 4):
                print("  " + ", ".join(missing_files[i:i+4]))
    else:
        print("❌ 未发现任何有效的 final_eval_metrics.csv 文件。")

if __name__ == "__main__":
    aggregate_results()