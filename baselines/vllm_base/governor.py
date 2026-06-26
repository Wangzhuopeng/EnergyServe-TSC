import subprocess

class VLLMBaseGovernor:
    """
    Governor for Vanilla vLLM baseline.
    Ensures the GPU always operates at peak power limit without any scaling.
    """
    def __init__(self, config):
        self.cfg = config
        # Default peak power (usually 400W for A800/H800)
        self.current_limit = 400 

    def adjust(self, running_reqs):
        """Vanilla vLLM does not perform dynamic power scaling."""
        pass

    def reset(self):
        """Ensure GPU is set to maximum power limit."""
        try:
            subprocess.run(["nvidia-smi", "-pl", str(self.current_limit)], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass