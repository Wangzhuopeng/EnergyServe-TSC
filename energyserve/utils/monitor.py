import threading
import time
import os
import pynvml
import pandas as pd

class RealEnergyMonitor(threading.Thread):
    """
    Background thread for monitoring GPU power consumption via NVML.
    """
    def __init__(self, gpu_id, interval=0.1, output_dir="outputs/results"):
        super().__init__()
        self.gpu_id = gpu_id
        self.interval = interval
        self.output_dir = output_dir
        self.dataset_name = "default"
        self.running = True
        self.energy_readings = []
        
        # Initialize NVML
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self.available = True
        except Exception as e:
            print(f"⚠️ Warning: NVML initialization failed: {e}")
            self.available = False

    def set_dataset_name(self, name):
        self.dataset_name = name

    def run(self):
        """Main monitoring loop."""
        if not self.available: return
        
        while self.running:
            try:
                # NVML returns power in milliwatts, convert to Watts
                power_mW = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                power_W = power_mW / 1000.0
                self.energy_readings.append({
                    "time": time.time(),
                    "power": power_W
                })
            except:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.running = False

    def save(self):
        """Save power trace to CSV."""
        if self.energy_readings:
            path = os.path.join(self.output_dir, f"energyserve_{self.dataset_name}_power.csv")
            pd.DataFrame(self.energy_readings).to_csv(path, index=False)
            print(f"💾 Saved power trace: {path}")

    def get_total_energy(self):
        """Calculate total energy consumption (Joules) via integration."""
        if not self.energy_readings: return 0.0
        # Energy (J) = Power (W) * Time (s)
        return sum(r['power'] for r in self.energy_readings) * self.interval