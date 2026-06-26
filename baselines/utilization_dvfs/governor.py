import time
import subprocess
import pynvml

class UtilizationDVFSGovernor:
    """
    Baseline Governor: Reactive Utilization-based DVFS.
    Adjusts power based on GPU utilization reported by NVML, ignoring LLM semantics.
    """
    def __init__(self, config):
        self.cfg = config
        self.p_high = 300   # W (Turbo)
        self.p_mid  = 210   # W (Transition)
        self.p_low  = 170   # W (Eco)
        
        self.u_high = 95    # Utilization High Threshold
        self.u_low  = 60    # Utilization Low Threshold
        
        self.current_limit = self.p_high
        self.last_check_time = 0
        self.check_interval = 0.5 # 500ms check interval
        self.cooldown = 0 # Cooldown counter to prevent oscillation

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
        except:
            self.available = False

    def step(self):
        """Reactive logic triggered in the scheduler loop."""
        if not self.available: return
        
        now = time.time()
        if now - self.last_check_time < self.check_interval:
            return
        
        self.last_check_time = now

        try:
            # 1. Get current GPU utilization percentage
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = util_rates.gpu
            
            target = self.current_limit

            # 2. Reactive Logic (Preserved from original script)
            if gpu_util > self.u_high:
                # High load -> Burst to peak and set long cooldown
                target = self.p_high
                self.cooldown = 30 
                
            elif gpu_util < self.u_low:
                # Low load -> Check if still in cooldown period
                if self.cooldown > 0:
                    self.cooldown -= 1
                    target = self.p_high # Maintain high power during cooldown
                else:
                    target = self.p_low # Safe to downscale
            else:
                # Medium load
                if self.cooldown > 0:
                    self.cooldown -= 1
                    target = self.p_high
                else:
                    target = self.p_mid

            # 3. Apply hardware constraint
            if target != self.current_limit:
                self._apply(target)
                
        except Exception:
            pass

    def _apply(self, limit):
        try:
            subprocess.run(["nvidia-smi", "-pl", str(int(limit))], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.current_limit = limit
        except: pass

    def reset(self):
        """Reset to high power state."""
        self._apply(self.p_high)