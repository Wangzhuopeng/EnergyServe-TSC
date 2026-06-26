import subprocess

class FastServeGovernor:
    """
    Governor for FastServe baseline. 
    Maintains peak power (Turbo mode) throughout the execution.
    """
    def __init__(self, config):
        self.cfg = config
        self.current_limit = 400 # Peak Power

    def adjust(self, running_reqs):
        # FastServe does not scale voltage/frequency
        pass

    def reset(self):
        """Ensure GPU is at full power."""
        try:
            subprocess.run(["nvidia-smi", "-pl", str(self.current_limit)], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except: pass