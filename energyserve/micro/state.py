"""Live batch-state characterization for the micro-governor.

Implements the state-vector construction of Algorithm 2 (lines 1-2):

    s_t = [N_active, rho_long]

where ``N_active`` is the current request concurrency and ``rho_long`` is the
saturation ratio of long (memory-bound) decoding tasks in the running batch.
This lightweight signature is what distinguishes a transient compute burst from
a persistent memory-bound window, and it is precisely the stability synthesized
by the EAPS macro-scheduler that makes the signature reliable enough to act on.
"""
from dataclasses import dataclass


@dataclass
class BatchState:
    """Instantaneous signature of the running batch (the state vector s_t)."""

    n_active: int            # number of in-flight requests
    concurrency_ratio: float  # n_active / max_concurrency  (batch fullness)
    long_ratio: float         # fraction of long, memory-bound decoding tasks


class BatchStateProbe:
    """Builds a :class:`BatchState` from the live running requests.

    A request is counted as *long* when its predicted output length reaches the
    configured ``long_task_threshold`` (the same threshold the macro-scheduler
    uses to aggregate decodes into stable windows).
    """

    def __init__(self, config):
        self.max_concurrency = config["system"]["max_concurrency"]
        self.long_threshold = config["scheduler"]["long_task_threshold"]

    def probe(self, running_reqs):
        """Return the current :class:`BatchState`, or ``None`` if the batch is empty."""
        n_active = len(running_reqs)
        if n_active == 0:
            return None

        concurrency_ratio = n_active / self.max_concurrency
        long_cnt = sum(1 for r in running_reqs if r.expected_out_len >= self.long_threshold)
        long_ratio = long_cnt / n_active

        return BatchState(
            n_active=n_active,
            concurrency_ratio=concurrency_ratio,
            long_ratio=long_ratio,
        )
