import asyncio
import time
from dataclasses import dataclass, field

@dataclass
class ServingRequest:
    """
    Represents a single inference request within the EnergyServe system.
    Holds all necessary metadata for scheduling and SLO calculation.
    """
    request_id: str
    prompt: str
    expected_out_len: int  # Predicted by the Perception Model
    arrival_time: float    # Absolute timestamp when the request arrived
    
    # Lifecycle Timestamps
    enqueue_time: float = 0.0
    start_exec_time: float = 0.0
    finish_time: float = 0.0
    
    # Async synchronization primitive to signal completion
    event: asyncio.Event = field(default_factory=asyncio.Event)
    
    def __post_init__(self):
        # Automatically set enqueue time if not provided
        if self.enqueue_time == 0.0:
            self.enqueue_time = time.time()