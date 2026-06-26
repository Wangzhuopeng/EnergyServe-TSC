import os
import json
import yaml
import argparse
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ==========================================
# 1. Helper Functions
# ==========================================
def format_gemma_prompt(item):
    """Constructs a Gemma-compatible prompt."""
    instruction = item.get("instruction", "").strip()
    input_text = item.get("input", "").strip()
    content = f"{instruction}\n{input_text}" if input_text else instruction
    return f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"

def get_category_label(in_len, out_len, cfg):
    """Assigns a category label (e.g., 'SL', 'MM') based on input/output lengths."""
    # Input Tag (S/M/L)
    if in_len < cfg['thres_in_s']: in_tag = 'S'
    elif in_len < cfg['thres_in_m']: in_tag = 'M'
    else: in_tag = 'L'
    # Output Tag (S/M/L)
    if out_len < cfg['thres_out_s']: out_tag = 'S'
    elif out_len < cfg['thres_out_m']: out_tag = 'M'
    else: out_tag = 'L'
    return in_tag + out_tag

def bucket_to_tokens(bucket_id, bucket_size):
    """Converts a bucket ID back to an estimated token count."""
    return int(bucket_id * bucket_size + (bucket_size / 2))

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 2. Parse Arguments and Load Config
    parser = argparse.ArgumentParser(description="Analyze workload distribution using the perception model")
    parser.add_argument("--config", type=str, default="configs/serving.yaml")
    parser.add_argument("--dataset", type=str, required=True, choices=["lmsys", "alpaca", "dolly"])
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config['datasets'][args.dataset]
    char_cfg = config['characterization']

    # 3. Load Data
    print(f">>> Loading test data: {dataset_cfg['data_path']} ...")
    prompts = []
    with open(dataset_cfg['data_path'], "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            prompts.append(format_gemma_prompt(json.loads(line)))
    print(f">>> Total samples loaded: {len(prompts)}")

    # 4. Initialize vLLM and Run Inference
    print(">>> Initializing vLLM to predict output lengths...")
    llm = LLM(
        model=config['hardware']['base_model'], enable_lora=True, 
        max_model_len=2048, max_lora_rank=128, gpu_memory_utilization=0.6,
        enforce_eager=True, trust_remote_code=True, dtype="bfloat16"
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=10, stop=["<end_of_turn>"])
    
    outputs = llm.generate(
        prompts, sampling_params,
        lora_request=LoRARequest("gemma_adapter", 1, dataset_cfg['lora_path'])
    )

    # 5. Process Results and Classify
    print("\n>>> Processing results and classifying requests...")
    category_counts = Counter()
    results_list = []
    
    for i, output in enumerate(outputs):
        input_len = len(output.prompt_token_ids)
        gen_text = output.outputs[0].text.strip()
        
        clean_text = ''.join(filter(str.isdigit, gen_text))
        bucket_id = int(clean_text) if clean_text else 0
        predicted_output_len = bucket_to_tokens(bucket_id, char_cfg['bucket_size'])
        
        label = get_category_label(input_len, predicted_output_len, char_cfg)
        category_counts[label] += 1
        
        results_list.append({
            "input_len": input_len, "pred_bucket": bucket_id,
            "pred_tokens": predicted_output_len, "label": label
        })

    # 6. Generate and Print Report
    total = len(results_list)
    category_order = ['SS', 'SM', 'SL', 'MS', 'MM', 'ML', 'LS', 'LM', 'LL']
    
    print("\n" + "="*50)
    print(f"📊 Predicted Workload Distribution ({args.dataset.upper()}, N={total})")
    print("="*50)
    print(f"{'Category':<10} | {'Count':<8} | {'Percentage':<10}")
    print("-" * 34)
    for cat in category_order:
        count = category_counts[cat]
        percent = (count / total) * 100
        print(f"{cat:<10} | {count:<8} | {percent:.2f}%")
    print("="*50)

    # 7. Key Insight: Long-tail Ratio
    long_tail_count = category_counts['SL'] + category_counts['ML'] + category_counts['LL']
    print(f"\n🔍 Key Insight:")
    print(f"Predicted ratio of long-output tasks (L >= {char_cfg['thres_out_m']}): {(long_tail_count/total)*100:.2f}%")
    
    # 8. Save results
    output_path = f"outputs/results/{args.dataset}_distribution.csv"
    pd.DataFrame(results_list).to_csv(output_path, index=False)
    print(f"💾 Detailed classification saved to: {output_path}")

if __name__ == "__main__":
    main()