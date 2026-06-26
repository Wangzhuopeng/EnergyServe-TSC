import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from energyserve.utils.analyzer import ExperimentAnalyzer

# Professional style for academic papers
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2})

class PaperPlotter:
    def __init__(self, dataset="lmsys"):
        self.dataset = dataset
        self.analyzer = ExperimentAnalyzer()
        self.modes = ["vllm_base", "fixed_150w", "utilization_dvfs", "fastserve", "dynamollm", "ours"]
        self.colors = ['gray', 'brown', 'orange', 'blue', 'purple', 'red']

    def plot_figure_7_power_profile(self):
        """Generates Figure 7: Raw Power Profiles (Bimodal behavior)."""
        plt.figure(figsize=(12, 6))
        
        # We plot Ours vs Base for clear contrast
        for mode in ["vllm_base", "ours"]:
            pwr_file = f"outputs/results/{mode}_{self.dataset}_power.csv"
            if os.path.exists(pwr_file):
                df = pd.read_csv(pwr_file)
                # Normalize time to start from 0
                df['time'] -= df['time'].iloc[0]
                label = "EnergyServe (Ours)" if mode == "ours" else "vLLM (Base)"
                color = "red" if mode == "ours" else "gray"
                plt.plot(df['time'], df['power'], label=label, color=color, alpha=0.8)

        plt.axhline(y=400, color='black', linestyle='--', label='Turbo (400W)')
        plt.axhline(y=180, color='green', linestyle='--', label='Eco (180W)')
        plt.xlabel("Time (s)")
        plt.ylabel("GPU Power (W)")
        plt.title(f"Figure 7: Power Profile Comparison ({self.dataset.upper()})")
        plt.legend(loc='upper right', ncol=2)
        plt.grid(True, alpha=0.3)
        plt.savefig("outputs/figures/figure7_power.png", dpi=300)
        print("✅ Figure 7 saved.")

    def plot_figure_6_latency_cdf(self):
        """Generates Figure 6: End-to-End Latency CDF."""
        plt.figure(figsize=(8, 6))
        
        for mode, color in zip(self.modes, self.colors):
            req_file = f"outputs/results/{mode}_{self.dataset}_requests.csv"
            if os.path.exists(req_file):
                df = pd.read_csv(req_file)
                sorted_lat = np.sort(df['latency'])
                yvals = np.arange(len(sorted_lat)) / float(len(sorted_lat) - 1)
                plt.plot(sorted_lat, yvals, label=mode, color=color)

        plt.xlabel("Latency (s)")
        plt.ylabel("CDF")
        plt.title("Figure 6: Latency Cumulative Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("outputs/figures/figure6_cdf.png", dpi=300)
        print("✅ Figure 6 saved.")

    def generate_table_2(self):
        """Calculates and prints metrics for Table 2 comparison."""
        results = []
        for mode in self.modes:
            m = self.analyzer.calculate_metrics(mode, self.dataset)
            if m: results.append(m)
        
        df_table = pd.DataFrame(results)
        print("\n" + "="*80)
        print(f"📊 Table 2: End-to-End Performance on {self.dataset.upper()}")
        print("="*80)
        print(df_table.to_string(index=False))
        df_table.to_csv(f"outputs/results/table2_summary_{self.dataset}.csv", index=False)

if __name__ == "__main__":
    # Ensure figure directory exists
    os.makedirs("outputs/figures", exist_ok=True)
    
    plotter = PaperPlotter(dataset="lmsys")
    plotter.generate_table_2()
    plotter.plot_figure_7_power_profile()
    plotter.plot_figure_6_latency_cdf()