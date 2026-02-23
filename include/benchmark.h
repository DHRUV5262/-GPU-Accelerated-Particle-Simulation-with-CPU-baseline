#pragma once

#include <string>
#include <vector>

/// Runs benchmarks for CPU, GPU naive, and GPU optimized simulations
/// across the given particle counts. Uses a fixed random seed for reproducibility.
///
/// \param particle_counts  Vector of particle counts (e.g., {1000, 10000, 50000, 100000})
/// \param steps            Number of simulation steps per run
/// \param csv_path         Output CSV file path (columns: N,cpu_ms,gpu_naive_ms,gpu_opt_ms)
void run_benchmarks(const std::vector<int>& particle_counts, int steps,
                   const std::string& csv_path);
