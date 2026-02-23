#!/usr/bin/env python3
"""Read benchmark CSV and plot CPU vs GPU variants (naive, SoA, fused, auto-tune)."""

import csv
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    candidates = [
        project_root / "benchmark_results.csv",
        project_root / "build" / "benchmark_results.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        print("CSV not found. Run the benchmark first: ./gpu_particle_benchmark")
        print("Looked in:", [str(p) for p in candidates])
        sys.exit(1)

    n_vals = []
    cpu_ms = []
    gpu_naive_ms = []
    gpu_opt_ms = []
    gpu_unfused_ms = []
    gpu_autotune_ms = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_vals.append(int(row["N"]))
            cpu_ms.append(float(row["cpu_ms"]))
            gpu_naive_ms.append(float(row["gpu_naive_ms"]))
            gpu_opt_ms.append(float(row["gpu_opt_ms"]))
            if "gpu_soa_unfused_ms" in row:
                gpu_unfused_ms.append(float(row["gpu_soa_unfused_ms"]))
            if "gpu_autotune_ms" in row:
                gpu_autotune_ms.append(float(row["gpu_autotune_ms"]))

    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, cpu_ms, "o-", label="CPU (single-threaded)", linewidth=2)
    plt.plot(n_vals, gpu_naive_ms, "s-", label="GPU naive (AoS)", linewidth=2)
    plt.plot(n_vals, gpu_opt_ms, "^-", label="GPU SoA fused (256)", linewidth=2)
    if gpu_unfused_ms:
        plt.plot(n_vals, gpu_unfused_ms, "d-", label="GPU SoA unfused (2 kernels)", linewidth=2)
    if gpu_autotune_ms:
        plt.plot(n_vals, gpu_autotune_ms, "v-", label="GPU SoA auto-tune (128/256/512)", linewidth=2)

    plt.xlabel("Particle count (N)")
    plt.ylabel("Time (ms)")
    plt.title("Particle Simulation: CPU vs GPU Variants (SoA, Fused, Auto-tune)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = project_root / "benchmark_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
