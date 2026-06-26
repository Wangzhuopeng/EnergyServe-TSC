"""Hardware power actuation for the micro-governor.

Implements the actuation path of Algorithm 2 (lines 9-10): a first-order
slew-rate filter that bounds the per-step power delta, followed by enforcement
of the limit through the native NVIDIA driver interface (``nvidia-smi -pl``).

Separating actuation from policy keeps the hardware side-effects in one place:
the slew-rate clip avoids the di/dt transients that abrupt voltage steps would
otherwise induce, and all driver calls fail soft so a hardware hiccup never
crashes the serving engine.
"""
import subprocess


class PowerActuator:
    """Applies smoothed power limits to the GPU via the driver interface."""

    def __init__(self, config):
        gov = config["governor"]
        self.step_size = gov["step_size"]      # max per-step slew (W)
        self.p_turbo = gov["power_turbo"]      # reset / peak power
        self.current_limit = self.p_turbo

    def _smooth(self, target):
        """First-order slew-rate clip toward ``target`` (Eq. 12)."""
        if self.current_limit > target:
            return max(target, self.current_limit - self.step_size)
        if self.current_limit < target:
            return min(target, self.current_limit + self.step_size)
        return self.current_limit

    def actuate(self, target):
        """Move one slew-bounded step toward ``target`` and apply it.

        Returns the (possibly unchanged) active power limit in watts.
        """
        new_limit = self._smooth(target)
        if new_limit != self.current_limit:
            self._apply_limit(new_limit)
            self.current_limit = new_limit
        return self.current_limit

    def _apply_limit(self, limit):
        """Set the GPU power limit via nvidia-smi (fails soft)."""
        try:
            subprocess.run(
                ["nvidia-smi", "-pl", str(int(limit))],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            # Never crash the engine on a hardware actuation error.
            print(f"Hardware Error: Failed to set power limit: {e}")

    def reset(self):
        """Restore the GPU to the peak (turbo) power envelope."""
        self._apply_limit(self.p_turbo)
        self.current_limit = self.p_turbo
