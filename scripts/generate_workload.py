import os
import json
import yaml
import argparse
import pandas as pd
import numpy as np

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 1. Parse Arguments and Load Config
    parser = argparse.ArgumentParser(description="Generate experiment workload from prediction results")
    parser.add_argument("--config", type=str, default="configs/serving.yaml")
    parser.add_argument("--dataset", type=str, required=True, choices=["lmsys", "alpaca", "dolly"])
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config['datasets'][args.dataset]
    sim_cfg = config['simulation']

    # Define paths
    pred_csv_path = f"outputs/results/{args.dataset}_distribution.csv"
    original_jsonl_path = dataset_cfg['data_path'] # Assuming the test data is the original source
    output_workload_path = f"data/workload/{args.dataset}_workload.jsonl"

    # 2. Load Prediction Results
    print("🔨 Merging prediction results with original prompts...")
    if not os.path.exists(pred_csv_path):
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv_path}. Please run analyze_workload_distribution.py first.")
    df_pred = pd.read_csv(pred_csv_path)
    print(f"   - Loaded prediction data: {len(df_pred)} requests")

    # 3. Load Original Prompts
    prompts = []
    with open(original_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            instruction = item.get("instruction", "").strip()
            input_text = item.get("input", "").strip()
            content = f"{instruction}\n{input_text}" if input_text else instruction
            prompts.append(content)
    print(f"   - Loaded original prompts: {len(prompts)} prompts")

    # 4. Generate Workload with Arrival Times
    workload = []
    current_time = 0.0
    
    # Iterate through prediction results to ensure order
    for idx, row in df_pred.iterrows():
        if idx >= len(prompts): continue
        
        # Simulate Poisson arrival process
        current_time += np.random.exponential(1.0 / sim_cfg['arrival_rate'])
        
        workload.append({
            "arrival_time": current_time,
            "prompt": prompts[idx],
            "output_len": int(row['pred_tokens']), # Use the predicted length for the task
            "label": row['label']
        })

    # 5. Save the final workload file
    with open(output_workload_path, 'w') as f:
        for item in workload:
            f.write(json.dumps(item) + "\n")
            
    print(f"\n✅ Simulation workload generated: {output_workload_path}")
    print(f"   - Total requests: {len(workload)}")
    long_task_ratio = len(df_pred[df_pred['label'].str.contains('L')]) / len(df_pred) * 100
    print(f"   - Long-task ratio in workload: {long_task_ratio:.2f}%")

if __name__ == "__main__":
    main()