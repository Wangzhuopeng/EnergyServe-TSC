import asyncio
import time
from vllm import SamplingParams
from .governor import UtilizationDVFSGovernor

class UtilizationDVFSScheduler:
    """
    Utilization-DVFS Baseline: FCFS scheduling combined with reactive power control.
    """
    def __init__(self, engine, recorder, config):
        self.engine = engine
        self.recorder = recorder
        self.cfg = config
        self.waiting = []
        self.running = []
        self.governor = UtilizationDVFSGovernor(config)
        self.done_submit = False
        self.done_event = asyncio.Event()

    async def add(self, req):
        # FCFS: Simply record arrival and append
        req['enqueue_time'] = time.time()
        self.waiting.append(req)

    async def loop(self, pbar):
        """Main coordination loop for FCFS + Reactive Power."""
        self.governor.reset()
        
        while True:
            # 1. Update power via utilization feedback
            self.governor.step()
            
            # 2. Logging status
            if self.recorder:
                self.recorder.log_status(
                    waiting=len(self.waiting),
                    running=len(self.running),
                    limit=self.governor.current_limit
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

            # 5. Admission (First-Come-First-Serve)
            while len(self.running) < 64 and self.waiting:
                r = self.waiting.pop(0)
                r["event"] = asyncio.Event()
                r['start_ts'] = time.time()
                asyncio.create_task(self.run_one(r))
                self.running.append(r)
            
            await asyncio.sleep(0.01)

    async def run_one(self, req):
        """Standard execution with Oracle SLO calculation."""
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{req['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        params = SamplingParams(temperature=0.7, max_tokens=2048)
        
        actual_len = 0
        try:
            async for o in self.engine.generate(prompt, params, req["id"]):
                if o.outputs and o.outputs[0].token_ids:
                    actual_len = max(actual_len, len(o.outputs[0].token_ids))
        except: pass
        
        finish_time = time.time()
        # Oracle SLO: 13 + actual_len * 3
        slo_miss = (finish_time - req['enqueue_time']) > (13 + actual_len * 3)
        
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