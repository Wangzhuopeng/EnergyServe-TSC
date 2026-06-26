import asyncio
import time
from vllm import SamplingParams

class EnergyServeEngine:
    """
    Core Engine implementing Algorithm 3: Cross-Level System Logic.
    
    It coordinates the three main components:
    1. Macro-Scheduler: For admission control (SJF).
    2. Micro-Governor: For adaptive power scaling (DVFS).
    3. vLLM Engine: For token generation.
    """
    def __init__(self, vllm_engine, scheduler, governor, logger, config):
        """
        Args:
            vllm_engine: Initialized AsyncLLMEngine.
            scheduler: Instance of EnergyServeScheduler.
            governor: Instance of EnergyServeGovernor.
            logger: DataLogger for metrics.
            config: Loaded 'core_config.yaml'.
        """
        self.engine = vllm_engine
        self.scheduler = scheduler
        self.governor = governor
        self.logger = logger
        self.cfg = config
        
        # Track currently executing requests for the Governor
        self.running_requests = []
        
        # Signal to stop the main loop
        self.done_event = asyncio.Event()

    async def main_loop(self, pbar):
        """
        The continuous control loop that runs throughout the experiment.
        """
        while True:
            # 1. Micro-Governance (Algorithm 2)
            # Adjust power based on the characteristics of running requests
            self.governor.adjust(self.running_requests)

            # 2. System Monitoring
            # Log current queue depth, concurrency, and hardware power limit
            if self.logger:
                self.logger.log_status(
                    len(self.scheduler.waiting_queue), 
                    len(self.running_requests), 
                    self.governor.current_limit
                )

            # 3. Lifecycle Management
            # Remove requests that have finished execution
            self.running_requests = [r for r in self.running_requests if not r.event.is_set()]
            
            # 4. Termination Check
            # Stop if no more requests are coming, queue is empty, and engine is idle
            if (self.scheduler.done_submit and 
                not self.scheduler.waiting_queue and 
                not self.running_requests):
                self.done_event.set()
                break

            # 5. Admission Control (Macro-Scheduling)
            # Admit requests from the sorted queue up to the physical concurrency limit
            max_concur = self.cfg['system']['max_concurrency']
            
            while len(self.running_requests) < max_concur and self.scheduler.waiting_queue:
                # Get the highest priority request (SJF logic handled inside scheduler)
                req = self.scheduler.pop_next()
                
                # Mark execution start
                req.start_exec_time = time.time()
                
                # Launch asynchronous inference
                asyncio.create_task(self._run_inference(req, pbar))
                
                # Register to running list
                self.running_requests.append(req)
            
            # Yield control to allow async tasks to progress
            await asyncio.sleep(0.01)

    async def _run_inference(self, req, pbar):
        """
        Executes a single request and handles SLO verification.
        """
        # Define sampling params (Temperature=0.7 from original script)
        params = SamplingParams(temperature=0.7, max_tokens=2048)
        
        actual_len = 0
        try:
            # Stream tokens from vLLM
            async for output in self.engine.generate(req.prompt, params, req.request_id):
                if output.outputs:
                    # Update actual length for Oracle SLO calculation
                    actual_len = len(output.outputs[0].token_ids)
        except Exception as e:
            print(f"Inference Error [ReqID: {req.request_id}]: {e}")
        
        req.finish_time = time.time()
        
        # Oracle SLO Calculation (Eq 4)
        # Deadline = Base + Generated_Tokens * Budget_Per_Token
        slo_base = self.cfg['scheduler']['slo_base']
        slo_token = self.cfg['scheduler']['slo_token']
        
        slo_deadline = slo_base + actual_len * slo_token
        latency = req.finish_time - req.enqueue_time
        slo_miss = latency > slo_deadline
        
        # Record metrics
        if self.logger:
            # Estimate input length (chars / 4)
            input_len = len(req.prompt) // 4 
            self.logger.log_request(
                req.request_id,
                req.arrival_time,
                req.start_exec_time,
                req.finish_time,
                input_len,
                actual_len,
                slo_miss
            )
        
        # Signal completion
        req.event.set()
        pbar.update(1)