import asyncio
import time
import subprocess
from vllm import SamplingParams

class VLLMBaseScheduler:
    """
    VLLM-Base Baseline: Implements standard First-Come-First-Serve (FCFS) scheduling.
    Maintains a simple FIFO queue without any priority or aging logic.
    """
    def __init__(self, engine, recorder, config):
        self.engine = engine
        self.recorder = recorder
        self.cfg = config
        
        self.waiting = [] 
        self.running = [] 
        self.done_submit = False
        self.done_event = asyncio.Event()
        
        # Max concurrency limit from config (usually 64)
        self.max_concur = 64

    async def add(self, req):
        """Add request to queue in FIFO order."""
        req['enqueue_time'] = time.time()
        self.waiting.append(req)

    async def loop(self, pbar):
        """Main coordination loop for FCFS scheduling."""
        # Baseline always runs at max power
        try:
            subprocess.run(["nvidia-smi", "-pl", "400"], stdout=subprocess.DEVNULL)
        except: pass

        while True:
            # 1. System status logging (Power limit is fixed at 400W)
            if self.recorder:
                self.recorder.log_status(
                    waiting=len(self.waiting),
                    running=len(self.running),
                    limit=400
                )

            # 2. Manage request lifecycle
            alive = []
            finished = 0
            for r in self.running:
                if r["event"].is_set(): finished += 1
                else: alive.append(r)
            self.running = alive
            
            if finished: 
                pbar.update(finished)

            # 3. Termination Check
            if self.done_submit and not self.waiting and not self.running:
                self.done_event.set()
                break

            # 4. Admission Control (Simple FIFO pop from head)
            while len(self.running) < self.max_concur and self.waiting:
                req = self.waiting.pop(0) 
                req["event"] = asyncio.Event()
                req['start_ts'] = time.time()
                asyncio.create_task(self.run_one(req))
                self.running.append(req)
            
            await asyncio.sleep(0.01)

    async def run_one(self, req):
        """Standard execution loop with Oracle SLO check."""
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{req['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        params = SamplingParams(temperature=0.7, max_tokens=2048)
        
        actual_len = 0
        try:
            # Asynchronous generation to get final output length
            async for o in self.engine.generate(prompt, params, req["id"]):
                if o.finished:
                    actual_len = len(o.outputs[0].token_ids)
        except Exception:
            pass
        
        end_time = time.time()
        latency = end_time - req['enqueue_time']
        
        # Oracle SLO Calculation: 13 + actual_len * 3
        slo_threshold = 13 + actual_len * 3
        slo_miss = latency > slo_threshold
        
        if self.recorder:
            # Log metrics using the unified recorder format
            self.recorder.log_request(
                req_id=req['id'],
                arrival_ts=req['enqueue_time'],
                start_ts=req.get('start_ts', req['enqueue_time']),
                end_ts=end_time,
                in_len=len(req['prompt']) // 4,
                out_len=actual_len,
                slo_miss=slo_miss
            )
        req["event"].set()