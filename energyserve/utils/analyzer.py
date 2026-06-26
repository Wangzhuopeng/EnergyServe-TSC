import pandas as pd
import numpy as np
import os

class ExperimentAnalyzer:
    """
    Core math engine to calculate paper metrics from raw CSV results.
    Computes: Latency (Avg/P50/P99), SLO Attainment, Throughput, Energy, and EDP.
    """
    def __init__(self, results_dir="outputs/results"):
        self.results_dir = results_dir

    def calculate_metrics(self, mode_name, dataset):
        """Calculates all Table 2 metrics for a specific experimental run."""
        req_file = os.path.join(self.results_dir, f"{mode_name}_{dataset}_requests.csv")
        pwr_file = os.path.join(self.results_dir, f"{mode_name}_{dataset}_power.csv")
        
        if not os.path.exists(req_file) or not os.path.exists(pwr_file):
            return None

        df_req = pd.read_csv(req_file)
        df_pwr = pd.read_csv(pwr_file)

        # 1. Latency Metrics
        avg_lat = df_req['latency'].mean()
        p50_lat = df_req['latency'].quantile(0.5)
        p99_lat = df_req['latency'].quantile(0.99)

        # 2. SLO Attainment (Oracle)
        slo_attainment = (1 - df_req['slo_miss'].mean()) * 100

        # 3. Throughput (TPS/RPS)
        duration = df_pwr['time'].iloc[-1] - df_pwr['time'].iloc[0]
        total_tokens = df_req['output_len'].sum()
        rps = len(df_req) / duration
        tps = total_tokens / duration

        # 4. Energy Integration (Eq 3)
        # Numerical integration using trapezoidal rule
        avg_power = df_pwr['power'].mean()
        total_energy = avg_power * duration 

        # 5. Energy-Delay Product (EDP) - The key metric
        # EDP = Total Energy * Average Latency
        edp = (total_energy * avg_lat) / 1e6 # Scaled for readability

        return {
            "mode": mode_name,
            "avg_lat": avg_lat, "p50_lat": p50_lat, "p99_lat": p99_lat,
            "slo": slo_attainment, "rps": rps, "tps": tps,
            "energy": total_energy, "edp": edp, "j_per_token": total_energy / total_tokens
        }