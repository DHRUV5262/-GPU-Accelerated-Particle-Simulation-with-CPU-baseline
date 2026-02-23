#include "benchmark.h"

#include <iostream>
#include <vector>

int main() {
    std::vector<int> particle_counts = {1'000, 10'000, 50'000, 100'000, 200'000, 300'000, 400'000};
    const int steps = 100;
    const std::string csv_path = "benchmark_results.csv";

    std::cout << "Running benchmarks (N = ";
    for (size_t i = 0; i < particle_counts.size(); ++i) {
        std::cout << particle_counts[i];
        if (i + 1 < particle_counts.size()) std::cout << ", ";
    }
    std::cout << ", steps = " << steps << ")...\n";

    run_benchmarks(particle_counts, steps, csv_path);

    std::cout << "Results written (see path above).\n";
    std::cout << "Run scripts/plot_results.py to generate the chart.\n";
    std::cout << "Press Enter to exit...";
    std::cin.get();

    return 0;
}
