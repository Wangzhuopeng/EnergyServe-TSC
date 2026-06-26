import asyncio
import yaml
import json
import argparse
import time
import uuid
import sys
import os
from tqdm import tqdm

# Add project root to python path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm import AsyncLLMEngine, AsyncEngineArgs
from energyserve.core.engine import EnergyServeEngine
from energyserve.core.request import ServingRequest
from energyserve.macro.scheduler import EnergyServeScheduler
from energyserve.micro.governor import EnergyServeGovernor
from energyserve.utils.logger import DataLogger
from energyserve.utils.monitor import RealEnergyMonitor

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

async def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Run EnergyServe Inference System")
    parser.add_argument("--dataset", type=str, required=True, choices=["lmsys", "alpaca", "dolly"], 
                        help="Which dataset workload to run")
    parser.add_argument("--config", type=str, default="configs/core_config.yaml",
                        help="Path to the system configuration file")
    args = parser.parse_args()

    # 2. Load Configuration
    cfg = load_config(args.config)
    print(f"🚀 Starting EnergyServe | Dataset: {args.dataset}")

    # 3. Initialize vLLM Engine
    print(f">>> Initializing vLLM with model: {cfg['system']['base_model_path']}")
    engine_args = AsyncEngineArgs(
        model=cfg['system']['base_model_path'],
        max_model_len=4096,
        max_num_seqs=cfg['system']['max_concurrency'],
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        disable_log_stats=True
    )
    vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    # 4. Initialize EnergyServe Components
    logger = DataLogger(dataset_name=args.dataset)
    monitor = RealEnergyMonitor(gpu_id=cfg['system']['gpu_id'])
    monitor.set_dataset_name(args.dataset)
    
    scheduler = EnergyServeScheduler(cfg)
    governor = EnergyServeGovernor(cfg)
    
    # 5. Build the System Engine
    system = EnergyServeEngine(vllm_engine, scheduler, governor, logger, cfg)

    # 6. Load Workload Data
    workload_path = cfg['workloads'][args.dataset]
    print(f">>> Loading workload from: {workload_path}")
    if not os.path.exists(workload_path):
        print(f"❌ Error: Workload file not found: {workload_path}")
        print("   Did you run scripts/generate_workload.py?")
        return

    workload_data = []
    with open(workload_path, 'r') as f:
        for line in f:
            if line.strip():
                workload_data.append(json.loads(line))
    
    print(f"📦 Total Requests: {len(workload_data)}")

    # 7. Start Monitoring and Main Loop
    monitor.start()
    pbar = tqdm(total=len(workload_data), desc="Serving")
    
    # Launch the system loop as a background task
    loop_task = asyncio.create_task(system.main_loop(pbar))

    # 8. Feed Requests (Simulate Arrival)
    start_time = time.time()
    time_compression = cfg['system']['time_compression']
    
    for i, data in enumerate(workload_data):
        # Calculate target arrival time
        now = time.time() - start_time
        target_time = data['arrival_time'] * time_compression
        
        if target_time > now:
            await asyncio.sleep(target_time - now)
            
        # Create Request Object
        req = ServingRequest(
            request_id=str(uuid.uuid4()),
            prompt=data['prompt'],
            expected_out_len=data['output_len'], # Prediction from perception layer
            arrival_time=time.time()
        )
        
        # Submit to Scheduler
        scheduler.add_request(req)

    # 9. Signal Completion and Cleanup
    print("\n>>> All requests submitted. Waiting for completion...")
    scheduler.done_submit = True
    
    # Wait for the engine to process all remaining requests
    await system.done_event.wait()
    
    # Stop background tasks
    loop_task.cancel()
    monitor.stop()
    
    # 10. Save Results
    print("\n>>> Saving results...")
    logger.save()
    monitor.save()
    
    # Calculate Total Energy
    total_energy = monitor.get_total_energy()
    print(f"\n✅ Experiment Complete!")
    print(f"⚡ Total Energy Consumed: {total_energy:.2f} Joules")
    
    # Reset GPU to default state
    governor.reset()

if __name__ == "__main__":
    asyncio.run(main())