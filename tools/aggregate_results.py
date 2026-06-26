"""
Aggregate EnergyServe serving results into a single comparison table.

Reads the per-run CSVs produced by ``run_serving.py`` / ``run_baselines.py``
(via ``energyserve.utils.logger.DataLogger`` and ``RealEnergyMonitor``):

    outputs/results/energyserve_<tag>_requests.csv   # per-request metrics
    outputs/results/energyserve_<tag>_power.csv      # power trace (time, power)

For every run it computes latency percentiles, Oracle-SLO attainment, total
energy (trapezoidal integration of the power trace), and the Energy-Delay
Product (EDP = total_energy x avg_latency), then writes a combined summary.

Usage:
    python tools/aggregate_results.py --results-dir outputs/results \
                                      --out outputs/summary.csv
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

# Oracle SLO deadline: base + output_len * per-token budget (paper Eq. 4).
SLO_BASE = 13
SLO_TOKEN = 3

REQ_RE = re.compile(r"energyserve_(?P<tag>.+)_requests\.csv$")


def integrate_energy(power_csv):
    """Total energy in Joules from a (time, power) trace via the trapezoid rule."""
    if not os.path.exists(power_csv):
        return np.nan
    df = pd.read_csv(power_csv)
    if df.empty or "time" not in df or "power" not in df:
        return np.nan
    t = df["time"].to_numpy()
    p = df["power"].to_numpy()
    if len(t) < 2:
        return float(p.sum())
    return float(np.trapz(p, t))


def summarize_run(tag, req_csv, results_dir):
    df = pd.read_csv(req_csv)
    if df.empty:
        return None

    lat = df["latency"]
    out_len = df["output_len"]
    # Recompute SLO from actual output length so every run uses the same rule.
    slo_miss = lat > (SLO_BASE + out_len * SLO_TOKEN)

    power_csv = os.path.join(results_dir, f"energyserve_{tag}_power.csv")
    energy = integrate_energy(power_csv)
    avg_lat = float(lat.mean())
    edp = energy * avg_lat if not np.isnan(energy) else np.nan

    total_tokens = float(out_len.sum())
    return {
        "Run": tag,
        "Requests": len(df),
        "Avg Lat (s)": avg_lat,
        "P50 Lat (s)": float(lat.quantile(0.50)),
        "P99 Lat (s)": float(lat.quantile(0.99)),
        "Oracle SLO (%)": float((1 - slo_miss.mean()) * 100),
        "Total E (J)": energy,
        "E/Req (J)": energy / len(df) if not np.isnan(energy) else np.nan,
        "E/Tok (J)": energy / total_tokens if (total_tokens and not np.isnan(energy)) else np.nan,
        "EDP (1e6)": edp / 1e6 if not np.isnan(edp) else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate EnergyServe results")
    parser.add_argument("--results-dir", default="outputs/results",
                        help="Directory containing energyserve_*_requests.csv files")
    parser.add_argument("--out", default="outputs/summary.csv",
                        help="Path to write the combined summary CSV")
    args = parser.parse_args()

    req_files = sorted(glob.glob(os.path.join(args.results_dir, "energyserve_*_requests.csv")))
    if not req_files:
        print(f"No result files found in {args.results_dir}")
        return

    rows = []
    for req_csv in req_files:
        m = REQ_RE.search(os.path.basename(req_csv))
        if not m:
            continue
        row = summarize_run(m.group("tag"), req_csv, args.results_dir)
        if row:
            rows.append(row)

    summary = pd.DataFrame(rows).set_index("Run").sort_index()
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("=" * 100)
    print("EnergyServe — aggregated results")
    print("=" * 100)
    print(summary.to_string())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summary.to_csv(args.out)
    print(f"\nSaved summary: {args.out}")


if __name__ == "__main__":
    main()
