import asyncio
import time
import subprocess
from vllm import SamplingParams

class FastServeScheduler:
    """
    FastServe Baseline: Implements Preemptive scheduling based on Input Tokens.
    Logic: If a waiting request is very short, preempt a running long task.
    """
    def __init__(self, engine, recorder, config):
        self.engine = engine
        self.recorder = recorder
        self.cfg = config
        
        # Thresholds from original code
        self.long_threshold = 300
        self.short_threshold = 100
        self.swap_overhead = 0.05 # 50ms swap cost
        self.max_concur = 64
        
        self.waiting = [] 
        self.running = [] 
        self.done_submit = False
        self.done_event = asyncio.Event()
        self.preempt_count = 0

    async def add(self, req):
        # req is a dictionary
        req['enqueue_time'] = time.time()
        # FastServe sorts by INPUT length (estimate: chars/4)
        req['input_tokens'] = len(req['prompt']) // 4
        self.waiting.append(req)
        self.resort_queue()

    def resort_queue(self):
        """Sort waiting queue by input length (FCFS with length priority)."""
        self.waiting.sort(key=lambda x: x["input_tokens"])

    async def loop(self, pbar):
        """Main coordination loop with preemption logic."""
        # Ensure peak power at start
        try:
            subprocess.run(["nvidia-smi", "-pl", "400"], stdout=subprocess.DEVNULL)
        except: pass

        while True:
            # 1. Logging system status
            if self.recorder:
                self.recorder.log_status(
                    waiting=len(self.waiting),
                    running=len(self.running),
                    limit=400
                )

            # 2. FastServe Preemption Logic
            if self.waiting and self.running:
                shortest_waiting = self.waiting[0]
                
                # Check if we should preempt a long running task to fit a short one
                if shortest_waiting["input_tokens"] < self.short_threshold:
                    victim = None
                    # Find a 'long' victim from the running pool
                    for r in reversed(self.running): 
                        if r["input_tokens"] > self.long_threshold:
                            victim = r
                            break
                    
                    if victim:
                        # Preempt the victim
                        victim["task"].cancel()
                        self.running.remove(victim)
                        # Simulate swap-out overhead
                        await asyncio.sleep(self.swap_overhead)
                        self.waiting.append(victim)
                        self.resort_queue()
                        self.preempt_count += 1

            # 3. Request lifecycle management
            alive = []
            finished = 0
            for r in self.running:
                if r["event"].is_set(): finished += 1
                else: alive.append(r)
            self.running = alive
            if finished: pbar.update(finished)

            # 4. Global termination
            if self.done_submit and not self.waiting and not self.running:
                self.done_event.set()
                break

            # 5. Admission (FCFS on the length-sorted queue)
            while len(self.running) < self.max_concur and self.waiting:
                req = self.waiting.pop(0) 
                req["event"] = asyncio.Event()
                req["start_ts"] = time.time()
                # Track the task object for possible preemption later
                task = asyncio.create_task(self.run_one(req))
                req["task"] = task 
                self.running.append(req)
            
            await asyncio.sleep(0.01)

    async def run_one(self, req):
        """Execution logic with support for task cancellation (preemption)."""
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{req['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        params = SamplingParams(temperature=0.7, max_tokens=2048)
        
        actual_len = 0
        try:
            async for o in self.engine.generate(prompt, params, req["id"]):
                if o.finished:
                    actual_len = len(o.outputs[0].token_ids)
        except asyncio.CancelledError:
            # Task was preempted by FastServe logic
            return 
        except Exception: pass
        
        end_time = time.time()
        latency = end_time - req['enqueue_time']
        
        # Oracle SLO Calculation (Based on output_len from workload)
        slo = 13 + req["output_len"] * 3
        slo_miss = latency > slo
        
        if self.recorder:
            self.recorder.log_request(
                req_id=req['id'],
                arrival_ts=req['enqueue_time'],
                start_ts=req.get('start_ts', req['enqueue_time']),
                end_ts=end_time,
                in_len=req['input_tokens'],
                out_len=actual_len,
                slo_miss=slo_miss
            )
        req["event"].set()