import asyncio
import time
from vllm import SamplingParams
from .governor import DynamoGovernor

class DynamoScheduler:
    def __init__(self, engine, recorder, config):
        self.engine = engine
        self.recorder = recorder
        self.waiting = []
        self.running = []
        self.power_ctrl = DynamoGovernor(config)
        self.done_submit = False
        self.done_event = asyncio.Event()

    async def add(self, req):
        # req is a dictionary as per your original implementation
        req['enqueue_time'] = time.time()
        self.waiting.append(req)

    async def loop(self, pbar):
        """Active coordination loop strictly following your implementation."""
        while True:
            # 1. Power adjustment
            self.power_ctrl.set_smart_limit(self.running)
            
            # 2. System status logging via your recorder
            if self.recorder:
                self.recorder.log_status(
                    waiting=len(self.waiting),
                    running=len(self.running),
                    limit=self.power_ctrl.current_limit
                )

            # 3. Manage finished requests
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

            # 5. Admission (FCFS for DynamoLLM)
            while len(self.running) < 64 and self.waiting:
                r = self.waiting.pop(0) # FCFS: pop from head
                r["event"] = asyncio.Event()
                r['start_ts'] = time.time()
                asyncio.create_task(self.run_one(r))
                self.running.append(r)
            
            await asyncio.sleep(0.01)

    async def run_one(self, req):
        """Streaming generation with Oracle SLO check."""
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{req['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        params = SamplingParams(temperature=0.7, max_tokens=2048)
        
        actual_len = 0
        try:
            # Streaming to update actual_len in real-time
            async for o in self.engine.generate(prompt, params, req["id"]):
                if o.outputs and o.outputs[0].token_ids:
                    actual_len = max(actual_len, len(o.outputs[0].token_ids))
        except Exception: pass
        
        end_time = time.time()
        latency = end_time - req['enqueue_time']
        
        # Oracle SLO (Eq 4)
        slo_miss = latency > (13 + actual_len * 3)
        
        if self.recorder:
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