import os
import json
import yaml
import argparse
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ==========================================
# 1. Prompt Formatting (Handles different dataset structures)
# ==========================================
def format_prompt(item, format_type):
    """
    Constructs a Gemma-compatible prompt based on the dataset type.
    """
    if format_type == "alpaca":
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        content = f"{instruction}\n{input_text}" if input_text else instruction
    elif format_type == "dolly":
        content = item.get("input", "").strip() # Dolly combines instruction and context
    else: # Default for LMSYS or other formats
        content = item.get("input", "") or item.get("instruction", "")

    return f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 2. Parse Arguments and Load Config
    parser = argparse.ArgumentParser(description="Evaluate EnergyServe's Perception Model")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--dataset", type=str, required=True, choices=["lmsys", "alpaca", "dolly"])
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_config = config['datasets'][args.dataset]
    
    # 3. Load Data
    print(f">>> Loading test data: {dataset_config['test_path']} ...")
    prompts = []
    ground_truths = []
    
    if not os.path.exists(dataset_config['test_path']):
        raise FileNotFoundError(f"Test data not found at: {dataset_config['test_path']}")
    if not os.path.exists(dataset_config['lora_path']):
        print(f"⚠️ Warning: LoRA path not found at: {dataset_config['lora_path']}. Please train the model first.")

    with open(dataset_config['test_path'], "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            prompts.append(format_prompt(item, dataset_config['prompt_format']))
            ground_truths.append(str(item["output"]).strip())

    print(f">>> Total test samples loaded: {len(prompts)}")

    # 4. Initialize vLLM Engine
    print(f">>> Initializing vLLM with base model: {config['model']['base_model']}...")
    llm = LLM(
        model=config['model']['base_model'],
        enable_lora=True, 
        max_model_len=config['model']['max_model_len'], 
        max_lora_rank=config['model']['max_lora_rank'], 
        gpu_memory_utilization=config['model']['gpu_memory_utilization'], 
        enforce_eager=True,
        trust_remote_code=True,
        dtype=config['model']['dtype']
    )

    sampling_params = SamplingParams(
        temperature=config['generation']['temperature'], 
        max_tokens=config['generation']['max_tokens'], 
        stop=config['generation']['stop_tokens']
    )

    # 5. Run Batch Inference
    print(">>> Starting batch inference...")
    try:
        outputs = llm.generate(
            prompts, 
            sampling_params,
            lora_request=LoRARequest("gemma_adapter", 1, dataset_config['lora_path'])
        )
    except Exception as e:
        print(f"❌ Inference runtime error: {e}")
        return

    # 6. Evaluate Results
    correct_count = 0
    fuzzy_correct_count = 0
    total_count = len(ground_truths)
    error_log = []

    print("\n>>> Calculating metrics...")
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        true_label_str = ground_truths[i]
        
        prediction_str = ''.join(filter(str.isdigit, generated_text.split()[0] if generated_text else ""))

        try:
            if not prediction_str: raise ValueError("Empty prediction")
            pred_val = int(prediction_str)
            true_val = int(true_label_str)
            
            if pred_val == true_val:
                correct_count += 1
            
            if abs(pred_val - true_val) <= config['metrics']['fuzzy_tolerance']:
                fuzzy_correct_count += 1
            else:
                if len(error_log) < 10:
                    error_log.append({
                        "label": true_val, "pred": pred_val, 
                        "diff": abs(pred_val - true_val), "raw": generated_text
                    })
        except ValueError:
            pass

    # 7. Print Report
    exact_acc = (correct_count / total_count) * 100
    fuzzy_acc = (fuzzy_correct_count / total_count) * 100
    
    print("\n" + "="*60)
    print(f"📊 Perception Model Evaluation Report ({args.dataset.upper()})")
    print("="*60)
    print(f"Total Samples:          {total_count}")
    print(f"Exact Matches:          {correct_count}")
    print(f"Fuzzy Matches (±{config['metrics']['fuzzy_tolerance']}):    {fuzzy_correct_count}")
    print("-" * 30)
    print(f"🎯 Exact Accuracy:         {exact_acc:.2f}%")
    print(f"🌊 Fuzzy Accuracy:         {fuzzy_acc:.2f}%")
    print("="*60)
    
    if error_log:
        print("\n❌ Error Analysis (Top 10):")
        print(f"{'Label':<10} | {'Pred':<10} | {'Diff':<10} | {'Raw Output'}")
        print("-" * 50)
        for err in error_log:
            print(f"{str(err['label']):<10} | {str(err['pred']):<10} | {str(err['diff']):<10} | {err['raw']}")

if __name__ == "__main__":
    main()