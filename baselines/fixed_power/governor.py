import subprocess

class FixedPowerGovernor:
    """
    Governor for Fixed-Power baseline. 
    Maintains a constant low power limit (e.g., 150W) regardless of workload.
    """
    def __init__(self, config):
        self.cfg = config
        # Fixed power limit defined in paper (150W)
        self.current_limit = 150 
        self._apply_fixed_limit()

    def adjust(self, running_reqs):
        """Fixed power strategy: No dynamic adjustment needed."""
        pass

    def _apply_fixed_limit(self):
        """Force apply the 150W limit using nvidia-smi."""
        try:
            subprocess.run(["nvidia-smi", "-pl", str(self.current_limit)], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Hardware Error: Failed to set fixed power limit: {e}")

    def reset(self):
        """Reset GPU to peak power (400W) after experiment."""
        try:
            subprocess.run(["nvidia-smi", "-pl", "400"], stdout=subprocess.DEVNULL)
        except: pass