"""
Run baseline serving policies under the same workload harness as EnergyServe.

Each baseline is a self-contained scheduler exposing a uniform interface:
    Scheduler(engine, recorder, config)
    await scheduler.add(req_dict)         # req has: id, prompt, output_len
    await scheduler.loop(pbar)            # main coordination loop
    scheduler.done_submit = True          # signal end of arrivals
    scheduler.done_event                  # asyncio.Event set on completion

Supported modes (paper baselines):
    vllm_base         FCFS @ peak power (Vanilla vLLM)
    fixed_power       FCFS under a static 150 W cap
    utilization_dvfs  FCFS + reactive utilization-based DVFS
    fastserve         Skip-join / shortest-input preemptive scheduling
    dynamollm         FCFS + phase-aware power policy

Usage:
    python scripts/run_baselines.py --mode dynamollm --dataset lmsys
"""
import asyncio
import json
import argparse
import time
import uuid
import sys
import os

import yaml
from tqdm import tqdm

# Add project root to python path so absolute imports resolve.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm import AsyncLLMEngine, AsyncEngineArgs
from energyserve.utils.logger import DataLogger
from energyserve.utils.monitor import RealEnergyMonitor

from baselines.vllm_base.scheduler import VLLMBaseScheduler
from baselines.vllm_base.governor import VLLMBaseGovernor
from baselines.fixed_power.scheduler import FixedPowerScheduler
from baselines.fixed_power.governor import FixedPowerGovernor
from baselines.utilization_dvfs.scheduler import UtilizationDVFSScheduler
from baselines.fastserve.scheduler import FastServeScheduler
from baselines.dynamollm.scheduler import DynamoScheduler

# mode -> (SchedulerClass, GovernorClass or None).
# A non-None GovernorClass is instantiated for its power side-effect / reset;
# dynamollm and utilization_dvfs own their governor internally.
REGISTRY = {
    "vllm_base":        (VLLMBaseScheduler, VLLMBaseGovernor),
    "fixed_power":      (FixedPowerScheduler, FixedPowerGovernor),
    "utilization_dvfs": (UtilizationDVFSScheduler, None),
    "fastserve":        (FastServeScheduler, None),
    "dynamollm":        (DynamoScheduler, None),
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def main():
    parser = argparse.ArgumentParser(description="Run EnergyServe baselines")
    parser.add_argument("--mode", type=str, required=True, choices=list(REGISTRY.keys()),
                        help="Which baseline policy to run")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["lmsys", "alpaca", "dolly"],
                        help="Which dataset workload to run")
    parser.add_argument("--config", type=str, default="configs/core_config.yaml",
                        help="Path to the system configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    SchedulerCls, GovernorCls = REGISTRY[args.mode]
    print(f"Starting baseline | mode: {args.mode} | dataset: {args.dataset}")

    # 1. Initialize vLLM engine (shared harness with run_serving.py).
    print(f">>> Initializing vLLM with model: {cfg['system']['base_model_path']}")
    engine_args = AsyncEngineArgs(
        model=cfg["system"]["base_model_path"],
        max_model_len=4096,
        max_num_seqs=cfg["system"]["max_concurrency"],
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    # 2. Recorder + energy monitor (tagged with the baseline mode).
    tag = f"{args.mode}_{args.dataset}"
    recorder = DataLogger(dataset_name=tag)
    monitor = RealEnergyMonitor(gpu_id=cfg["system"]["gpu_id"])
    monitor.set_dataset_name(tag)

    # 3. Build the baseline scheduler. Governors that act through a side-effect
    #    (fixed power cap) or need an explicit reset are instantiated here;
    #    dynamollm / utilization_dvfs manage their governor internally.
    scheduler = SchedulerCls(vllm_engine, recorder, cfg)
    governor = GovernorCls(cfg) if GovernorCls else None

    # 4. Load the workload.
    workload_path = cfg["workloads"][args.dataset]
    print(f">>> Loading workload from: {workload_path}")
    if not os.path.exists(workload_path):
        print(f"Error: workload file not found: {workload_path}")
        print("   Did you run scripts/generate_workload.py?")
        return

    workload_data = []
    with open(workload_path, "r") as f:
        for line in f:
            if line.strip():
                workload_data.append(json.loads(line))
    print(f"Total requests: {len(workload_data)}")

    # 5. Start monitoring and launch the scheduler loop.
    monitor.start()
    pbar = tqdm(total=len(workload_data), desc=f"Serving [{args.mode}]")
    loop_task = asyncio.create_task(scheduler.loop(pbar))

    # 6. Feed requests following the recorded Poisson arrival times.
    start_time = time.time()
    time_compression = cfg["system"]["time_compression"]
    for data in workload_data:
        now = time.time() - start_time
        target_time = data["arrival_time"] * time_compression
        if target_time > now:
            await asyncio.sleep(target_time - now)
        req = {
            "id": str(uuid.uuid4()),
            "prompt": data["prompt"],
            "output_len": data["output_len"],
        }
        await scheduler.add(req)

    # 7. Signal completion and wait for the queue to drain.
    print("\n>>> All requests submitted. Waiting for completion...")
    scheduler.done_submit = True
    await scheduler.done_event.wait()

    loop_task.cancel()
    monitor.stop()

    # 8. Persist results and restore the GPU to peak power.
    print("\n>>> Saving results...")
    recorder.save()
    monitor.save()
    total_energy = monitor.get_total_energy()
    print(f"\nExperiment complete! Total energy consumed: {total_energy:.2f} J")

    if governor is not None and hasattr(governor, "reset"):
        governor.reset()
    elif hasattr(scheduler, "governor") and hasattr(scheduler.governor, "reset"):
        scheduler.governor.reset()
    elif hasattr(scheduler, "power_ctrl") and hasattr(scheduler.power_ctrl, "reset"):
        scheduler.power_ctrl.reset()


if __name__ == "__main__":
    asyncio.run(main())
