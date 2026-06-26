import time
import os
import pandas as pd

class DataLogger:
    """
    Handles data collection for experimental analysis.
    Logs two types of data:
    1. System Status (Time-series data: Queue length, Power limit, Concurrency).
    2. Request Metrics (Per-request data: Latency, Input/Output length, SLO status).
    """
    def __init__(self, dataset_name, output_dir="outputs/results"):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.status_logs = []
        self.req_logs = []
        self.start_time = time.time()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def log_status(self, waiting, running, limit):
        """Snapshots the system state.

        Accepts both the positional calls used by the EnergyServe engine and
        the keyword calls (waiting=, running=, limit=) used by the baselines.
        """
        self.status_logs.append({
            "time": time.time(),
            "relative_time": time.time() - self.start_time,
            "waiting_len": waiting,
            "running_len": running,
            "power_limit": limit
        })

    def log_request(self, req_id, arrival_ts, start_ts, end_ts, in_len, out_len, slo_miss):
        """Records metrics for a completed request."""
        self.req_logs.append({
            "req_id": req_id,
            "arrival_time": arrival_ts,
            "start_time": start_ts,
            "end_time": end_ts,
            "latency": end_ts - arrival_ts,
            "wait_time": start_ts - arrival_ts,
            "exec_time": end_ts - start_ts,
            "input_len": in_len,
            "output_len": out_len,
            "slo_miss": slo_miss
        })

    def save(self):
        """Writes buffered logs to CSV files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if self.status_logs:
            path = os.path.join(self.output_dir, f"energyserve_{self.dataset_name}_status.csv")
            pd.DataFrame(self.status_logs).to_csv(path, index=False)
            print(f"💾 Saved system status log: {path}")
        
        if self.req_logs:
            path = os.path.join(self.output_dir, f"energyserve_{self.dataset_name}_requests.csv")
            pd.DataFrame(self.req_logs).to_csv(path, index=False)
            print(f"💾 Saved request metrics log: {path}")