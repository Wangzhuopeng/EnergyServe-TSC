import time
import asyncio

class EnergyServeScheduler:
    """
    Macro-Scheduler implementing the Starvation-Free Shortest Job First (SJF) algorithm.
    
    Paper Reference: Section 3.3, Algorithm 1.
    Core Logic: Reshapes the request queue to create stable execution phases while 
    preventing long-tail latency degradation via an aging mechanism.
    """
    def __init__(self, config):
        """
        Args:
            config (dict): The loaded 'core_config.yaml' dictionary.
        """
        self.cfg = config
        # Load the aging factor from config (e.g., 50.0)
        self.aging_factor = config['scheduler']['aging_factor']
        
        self.waiting_queue = []
        
        # Flags to coordinate system shutdown
        self.done_submit = False
        self.done_event = asyncio.Event()

    def add_request(self, req):
        """
        Admit a new request into the scheduling queue.
        
        Args:
            req (ServingRequest): The request object containing 'expected_out_len'.
        """
        # 1. Timestamp the entry for aging calculation
        req.enqueue_time = time.time()
        
        # 2. Add to queue
        self.waiting_queue.append(req)
        
        # 3. Trigger workload reshaping (Sort)
        self.resort_queue()

    def resort_queue(self):
        """
        The core SJF + Aging sorting logic (Eq 8 in the paper).
        
        Priority Score = Expected_Cost - (Waiting_Time * Aging_Factor)
        
        - Expected_Cost: The predicted output length from Perception Model.
        - Waiting_Time: Current Time - Enqueue Time.
        - Aging_Factor: Weight to balance efficiency (throughput) and fairness.
        
        Result: Short jobs go first, but old jobs eventually bubble up.
        """
        if not self.waiting_queue:
            return

        now = time.time()
        
        # Sort in-place. Python's sort is stable (Timsort).
        # Lower score = Higher Priority (Executed first).
        self.waiting_queue.sort(
            key=lambda x: x.expected_out_len - (now - x.enqueue_time) * self.aging_factor
        )

    def pop_next(self):
        """
        Retrieve the highest priority request.
        """
        if self.waiting_queue:
            return self.waiting_queue.pop(0)
        return None

    def has_pending_requests(self):
        """Check if the queue is empty."""
        return len(self.waiting_queue) > 0