import time
import asyncio

from .policy import EAPSPolicy

class EnergyServeScheduler:
    """
    Macro-Scheduler implementing the EAPS workload-reshaping policy (Algorithm 1).

    Paper Reference: Section 4.2, Algorithm 1.
    Core Logic: Reshapes the request queue to create stable execution phases while
    preventing long-tail latency degradation via an aging mechanism. The priority
    score itself is delegated to :class:`~energyserve.macro.policy.EAPSPolicy`
    (Eq. 8); this class owns only the queue mechanics (admission, sorting, pop).
    """
    def __init__(self, config):
        """
        Args:
            config (dict): The loaded 'core_config.yaml' dictionary.
        """
        self.cfg = config
        # EAPS priority score (expected cost - beta * aging); Eq. 8.
        self.policy = EAPSPolicy(config)

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
        Reshape the queue by EAPS priority (Eq. 8); lower score == higher priority.

        The per-request score (expected cost minus aging decay) is computed by
        :class:`~energyserve.macro.policy.EAPSPolicy`. Result: short jobs go
        first, but long-waiting jobs eventually bubble up (anti-starvation).
        """
        if not self.waiting_queue:
            return

        now = time.time()

        # Sort in-place. Python's sort is stable (Timsort).
        self.waiting_queue.sort(key=lambda r: self.policy.score(r, now))

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