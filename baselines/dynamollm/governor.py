import time
import subprocess

class DynamoGovernor:
    """
    Aggressive Energy-First Strategy strictly preserved from original code.
    Decision based on input_ratio, concurrency_ratio, and at_risk (Slack Check).
    """
    def __init__(self, config):
        self.cfg = config
        self.p_high = 250   # W (Turbo)
        self.p_mid = 200    # W (Panic/SLO Risk)
        self.p_low = 180    # W (Aggressive Eco)
        self.current_limit = self.p_high
        self.last_step_time = time.time()
        self.step_size = 50   
        self.step_interval = 0.1 

    def set_smart_limit(self, running_reqs):
        now = time.time()
        if now - self.last_step_time < self.step_interval:
            return

        total_running = len(running_reqs)
        target_goal = self.p_low 

        if total_running > 0:
            # 1. Feature extraction
            long_input_cnt = sum(1 for r in running_reqs if len(r['prompt']) > 1500) 
            input_ratio = long_input_cnt / total_running
            concurrency_ratio = total_running / 64 # PHYSICAL_MAX_CONCURRENCY

            # 2. SLO risk assessment (Slack Check)
            at_risk = False
            for r in running_reqs:
                elapsed = now - r['enqueue_time']
                # Oracle SLO threshold: 13 + output_len * 3
                expected_slo = 13 + r['output_len'] * 3
                if elapsed > (expected_slo * 0.7):
                    at_risk = True
                    break

            # 3. Decision tiers
            if input_ratio > 0.8 or concurrency_ratio > 0.98:
                target_goal = self.p_high  
            elif at_risk:
                target_goal = self.p_mid 
            else:
                target_goal = self.p_low 

        # 4. Rapid jump actuation logic
        new_limit = self.current_limit
        if abs(target_goal - self.current_limit) >= self.step_size:
            new_limit = target_goal
        else:
            if self.current_limit > target_goal:
                new_limit = max(target_goal, self.current_limit - self.step_size)
            elif self.current_limit < target_goal:
                new_limit = min(target_goal, self.current_limit + self.step_size)

        if new_limit != self.current_limit:
            try:
                subprocess.run(["nvidia-smi", "-pl", str(new_limit)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.current_limit = new_limit
                self.last_step_time = now
            except: pass

    def reset(self):
        subprocess.run(["nvidia-smi", "-pl", "400"], stderr=subprocess.DEVNULL)