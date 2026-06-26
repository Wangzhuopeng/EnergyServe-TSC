import time
import subprocess

class EnergyServeGovernor:
    """
    Micro-Governor implementing Algorithm 2: Adaptive DVFS Control.
    
    Responsibilities:
    1. Monitor real-time workload characteristics (Phase detection).
    2. Determine optimal power limit based on the three-tier strategy (A/B/C).
    3. Apply power constraints via hardware interface (nvidia-smi) with smoothing.
    """
    def __init__(self, config):
        """
        Args:
            config (dict): The loaded 'core_config.yaml'.
        """
        self.sys_cfg = config['system']
        self.sched_cfg = config['scheduler']
        self.gov_cfg = config['governor']
        
        # Initialize at Turbo (High Performance) state
        self.current_limit = self.gov_cfg['power_turbo']
        self.last_step_time = time.time()

    def adjust(self, running_reqs):
        """
        Main control loop triggered at each step.
        Calculates metrics and adjusts GPU power limit if necessary.
        
        Args:
            running_reqs (list): List of currently executing request objects.
        """
        # 0. Check Feature Flag
        if not self.gov_cfg['enable']:
            return

        # 1. Frequency Control (e.g., every 0.6s)
        now = time.time()
        if now - self.last_step_time < self.gov_cfg['step_interval']:
            return

        total_running = len(running_reqs)
        if total_running == 0:
            return

        # 2. Calculate Real-time Metrics (Phase Detection)
        max_concur = self.sys_cfg['max_concurrency']
        long_threshold = self.sched_cfg['long_task_threshold']
        
        # Concurrency Ratio: How full is the batch?
        concurrency_ratio = total_running / max_concur
        
        # Long Task Ratio: Is this a memory-bound decoding phase?
        long_cnt = sum(1 for r in running_reqs if r.expected_out_len >= long_threshold)
        long_ratio = long_cnt / total_running
        
        # 3. Determine Target Power Goal (Three-Tier Strategy)
        # Default: Turbo Mode (Compute-bound Prefill / Mixed)
        target_goal = self.gov_cfg['power_turbo']
        
        # Strategy C: Harvesting Phase (Stable Memory-bound)
        # Trigger: High ratio of long decoding tasks
        if long_ratio > self.gov_cfg['ratio_threshold_harvest']:
            target_goal = self.gov_cfg['power_eco'] # e.g., 180W
            
        # Strategy B: Idle Phase (Near-empty queue)
        # Trigger: Very low concurrency
        elif total_running < 2:
            target_goal = self.gov_cfg['power_idle'] # e.g., 150W
            
        # Strategy A: Stability Phase (High Concurrency)
        # Trigger: Batch is nearly full, lower voltage to reduce heat/jitter
        elif concurrency_ratio > self.gov_cfg['ratio_threshold_stable']:
            target_goal = self.gov_cfg['power_stable'] # e.g., 220W

        # 4. Smooth Actuation (Gradient Descent)
        # Avoid sudden voltage spikes (dI/dt noise) by moving in small steps
        step_size = self.gov_cfg['step_size']
        new_limit = self.current_limit
        
        if self.current_limit > target_goal:
            new_limit = max(target_goal, self.current_limit - step_size)
        elif self.current_limit < target_goal:
            new_limit = min(target_goal, self.current_limit + step_size)

        # 5. Apply Hardware Constraint
        if new_limit != self.current_limit:
            self._apply_limit(new_limit)
            self.current_limit = new_limit
            self.last_step_time = now

    def _apply_limit(self, limit):
        """Invoke nvidia-smi to set the power limit."""
        try:
            subprocess.run(
                ["nvidia-smi", "-pl", str(int(limit))], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            # Log error but don't crash the engine
            print(f"Hardware Error: Failed to set power limit: {e}")

    def reset(self):
        """Reset GPU to default high performance mode."""
        self._apply_limit(self.gov_cfg['power_turbo'])