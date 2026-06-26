"""Three-tier power policy for the micro-governor.

Implements the control law of Algorithm 2 (lines 3-8) / Eq. (11): a mapping from
the live batch state to a target power envelope. The policy partitions execution
into three regimes, prioritized as follows:

    Harvesting (Strategy C):  rho_long  > tau_harvest  -> P_eco
        A stable, memory-bound decoding window. Voltage is scaled down to
        saturate DRAM bandwidth without wasting dynamic power.
    Idle       (Strategy B):  N_active  < N_min        -> P_idle
        Near-empty batch; drop to the idle envelope to save energy.
    Stability  (Strategy A):  concurrency > tau_stable -> P_stable
        A nearly full batch; lower voltage moderately to curb heat and jitter.
    Performance (default):                              -> P_turbo
        Compute-bound prefill or mixed phase; hold peak power for latency.

The policy is pure: it reads a :class:`~energyserve.micro.state.BatchState` and
returns a target power level in watts, leaving smoothing and actuation to the
:mod:`~energyserve.micro.actuator`.
"""


class ThreeTierPowerPolicy:
    """Maps a :class:`BatchState` to a target power envelope (watts)."""

    # Minimum concurrency below which the batch is considered idle.
    IDLE_CONCURRENCY = 2

    def __init__(self, config):
        gov = config["governor"]
        self.p_turbo = gov["power_turbo"]      # performance / compute-bound
        self.p_eco = gov["power_eco"]          # harvesting (memory-bound window)
        self.p_idle = gov["power_idle"]        # idle (near-empty batch)
        self.p_stable = gov["power_stable"]    # stability (full batch)
        self.tau_harvest = gov["ratio_threshold_harvest"]
        self.tau_stable = gov["ratio_threshold_stable"]

    def target_power(self, state):
        """Return the target power limit (W) for the given batch state."""
        # Strategy C: harvesting dominates when long decodes saturate the batch.
        if state.long_ratio > self.tau_harvest:
            return self.p_eco

        # Strategy B: idle when very few requests are in flight.
        if state.n_active < self.IDLE_CONCURRENCY:
            return self.p_idle

        # Strategy A: stability when the batch is nearly full.
        if state.concurrency_ratio > self.tau_stable:
            return self.p_stable

        # Default: performance state to protect tail latency.
        return self.p_turbo
