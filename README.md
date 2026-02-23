# gpu-particle-benchmark

A C++17 + CUDA particle simulation comparing CPU baseline, GPU naive (AoS), and GPU optimized (SoA) implementations. Designed as a portfolio project for GPU software engineering roles (e.g., AMD SDE GPU) and easy to port to HIP/ROCm.

## What it does

Simulates many particles under gravity and a spring force toward the origin using semi-implicit Euler integration. Five GPU variants are implemented:

- **CPU baseline**: Single-threaded C++, Structure-of-Arrays (SoA) layout.
- **GPU naive (AoS)**: CUDA kernel with Array-of-Structures layout.
- **GPU SoA fused**: CUDA kernel with SoA layout, force+integration in one kernel, block size 256.
- **GPU SoA unfused**: SoA but two kernels (force compute, then integrate) for comparison.
- **GPU SoA auto-tune**: Same fused kernel, tries block sizes 128/256/512 and reports the best time.

A benchmark harness runs all five variants for multiple particle counts (1K, 10K, 50K, 100K) and writes timing results to CSV. A Python script plots the results.

## Build and run

Requirements: CMake 3.18+, C++17 compiler, CUDA toolkit.

```bash
mkdir build && cd build
cmake ..
cmake --build .
./gpu_particle_benchmark
```

Benchmark results are written to `benchmark_results.csv`. To plot:

```bash
pip install matplotlib
python scripts/plot_results.py
```

The plot is saved as `benchmark_plot.png`.

## Visualizer (optional)

A 2D real-time visualizer shows particle positions (x,y) and **FPS** so you can compare performance of each approach live.

**Requirements:** SDL2 and SDL2_ttf (e.g. `vcpkg install sdl2 sdl2-ttf`, or system packages on Linux).

If found at configure time, the `gpu_particle_visualizer` target is built. Run it from the build directory:

```bash
./gpu_particle_visualizer   # or .\Release\gpu_particle_visualizer.exe on Windows
```

- **FPS** and current **Method** are shown in the top-left.
- **Keys 1–5** switch approach: `1` CPU, `2` GPU Naive, `3` GPU SoA Fused (256), `4` GPU SoA Unfused, `5` GPU SoA Fused (512).
- **Escape** quits.

Particle count is fixed at 50,000 in the visualizer. Compare FPS when switching methods to see the impact of each optimization.

## Naive vs optimized kernels

| Variant | Layout | Kernels | Block size |
|---------|--------|---------|------------|
| Naive (AoS) | `struct Particle` | 1 | 256 |
| SoA fused | Separate arrays | 1 (force + integrate) | 256 |
| SoA unfused | Separate arrays | 2 (force, then integrate) | 256 |
| SoA auto-tune | Separate arrays | 1 (fused) | Best of 128/256/512 |

SoA improves GPU performance because adjacent threads in a warp access adjacent addresses, enabling efficient memory coalescing.

## Relevance to AMD GPU SDE roles

- **GPU programming**: Practical CUDA usage (kernels, launch configs, device memory).
- **Performance tuning**: Layout choice (AoS vs SoA), block size, and memory access patterns.
- **Portability**: C++ core and separation of CPU/GPU code make it straightforward to port kernels to HIP/ROCm for AMD GPUs.
- **Measurement**: Benchmarking with CUDA events and CSV output as a basis for optimization and regression checks.

## Project structure

```
CMakeLists.txt
include/
  particle_types.h      # Particle struct, constants
  cpu_sim.h
  gpu_sim_naive.h
  gpu_sim_optimized.h
  gpu_sim_soa_unfused.h
  benchmark.h
src/
  cpu_sim.cpp
  gpu_sim_naive.cu
  gpu_sim_optimized.cu
  gpu_sim_soa_unfused.cu
  benchmark.cpp
  main.cpp
  visualizer_main.cpp   # optional (SDL2 + SDL2_ttf)
scripts/
  plot_results.py
README.md
```
