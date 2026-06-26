"""Micro-Governor: the token-step DVFS controller (Algorithm 2).

This module is the orchestrator of the micro layer. It composes three
single-responsibility components, each mapping to a part of Algorithm 2:

    BatchStateProbe  (state.py)     -> build s_t            (lines 1-2)
    ThreeTierPowerPolicy (policy.py)-> choose target power  (lines 3-8, Eq. 11)
    PowerActuator    (actuator.py)  -> smooth + apply        (lines 9-10, Eq. 12)

The governor itself only handles the control cadence (rate gating) and wires the
components together, so the perception/decision/actuation concerns stay isolated
and independently testable.
"""
import time

from .state import BatchStateProbe
from .policy import ThreeTierPowerPolicy
from .actuator import PowerActuator


class EnergyServeGovernor:
    """Event-driven adaptive-DVFS governor (Algorithm 2)."""

    def __init__(self, config):
        """
        Args:
            config (dict): The loaded 'core_config.yaml'.
        """
        self.gov_cfg = config["governor"]

        # Single-responsibility components (perception -> decision -> actuation).
        self.probe = BatchStateProbe(config)
        self.policy = ThreeTierPowerPolicy(config)
        self.actuator = PowerActuator(config)

        self.last_step_time = time.time()

    @property
    def current_limit(self):
        """Active GPU power limit (W); read by the engine's status logger."""
        return self.actuator.current_limit

    def adjust(self, running_reqs):
        """Control step: profile the live batch and retune the power limit.

        Triggered once per token-generation step by the engine. Rate-gated to
        the configured ``step_interval`` to avoid di/dt churn.

        Args:
            running_reqs (list): Currently executing request objects.
        """
        # 0. Feature flag.
        if not self.gov_cfg["enable"]:
            return

        # 1. Control cadence: act at most once per step_interval.
        now = time.time()
        if now - self.last_step_time < self.gov_cfg["step_interval"]:
            return

        # 2. Perception: build the live batch state s_t (Alg. 2, lines 1-2).
        state = self.probe.probe(running_reqs)
        if state is None:
            return

        # 3. Decision: three-tier power policy (Alg. 2, lines 3-8).
        target = self.policy.target_power(state)

        # 4. Actuation: slew-rate smoothing + driver enforcement (lines 9-10).
        self.actuator.actuate(target)
        self.last_step_time = now

    def reset(self):
        """Restore the GPU to the default high-performance state."""
        self.actuator.reset()
