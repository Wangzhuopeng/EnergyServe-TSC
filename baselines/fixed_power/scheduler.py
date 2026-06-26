import asyncio
import time
from vllm import SamplingParams

class FixedPowerScheduler:
    """
    Fixed-Power Baseline: Implements FCFS scheduling under a strict power cap.
    Used to demonstrate the latency impact of non-adaptive power saving.
    """
    def __init__(self, engine, recorder, config):
        self.engine = engine
        self.recorder = recorder
        self.cfg = config
        self.waiting = []
        self.running = []
        self.done_submit = False
        self.done_event = asyncio.Event()
        
        # Max concurrency from config
        self.max_concur = 64
        self.fixed_limit = 150

    async def add(self, req):
        """Add request to queue in arrival order."""
        req['enqueue_time'] = time.time()
        self.waiting.append(req)

    async def loop(self, pbar):
        """Main coordination loop for FCFS + Fixed Power."""
        while True:
            # 1. System status logging (Power is fixed at 150W)
            if self.recorder:
                self.recorder.log_status(
                    waiting=len(self.waiting),
                    running=len(self.running),
                    limit=self.fixed_limit
                )

            # 2. Manage request cleanup
            alive = []
            finished = 0
            for r in self.running:
                if r["event"].is_set(): finished += 1
                else: alive.append(r)
            self.running = alive
            if finished: pbar.update(finished)

            # 3. Termination Check
            if self.done_submit and not self.waiting and not self.running:
                self.done_event.set()
                break

            # 4. Admission Control (FIFO)
            while len(self.running) < self.max_concur and self.waiting:
                req = self.waiting.pop(0) 
                req["event"] = asyncio.Event()
                req['start_ts'] = time.time()
                asyncio.create_task(self.run_one(req))
                self.running.append(req)
            
            await asyncio.sleep(0.01)

    async def run_one(self, req):
        """Standard execution with Oracle SLO check."""
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{req['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        params = SamplingParams(temperature=0.7, max_tokens=2048)
        
        actual_len = 0
        try:
            async for o in self.engine.generate(prompt, params, req["id"]):
                if o.finished:
                    actual_len = len(o.outputs[0].token_ids)
        except Exception: pass
        
        finish_time = time.time()
        latency = finish_time - req['enqueue_time']
        
        # Oracle SLO: 13 + actual_len * 3
        slo_miss = latency > (13 + actual_len * 3)
        
        if self.recorder:
            self.recorder.log_request(
                req_id=req['id'],
                arrival_ts=req['enqueue_time'],
                start_ts=req.get('start_ts', req['enqueue_time']),
                end_ts=finish_time,
                in_len=len(req['prompt']) // 4,
                out_len=actual_len,
                slo_miss=slo_miss
            )
        req["event"].set()