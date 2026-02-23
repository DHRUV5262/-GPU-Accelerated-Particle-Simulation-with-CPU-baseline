# gpu-particle-benchmark

A C++17 + CUDA particle simulation comparing CPU baseline, GPU naive (AoS), and GPU optimized (SoA) implementations. A learning project for exploring GPU programming and performance tuning.

## What it does

Simulates many particles under a spring force toward the origin using semi-implicit Euler integration. Five variants are implemented:

- **CPU baseline**: Single-threaded C++, Structure-of-Arrays (SoA) layout.
- **GPU naive (AoS)**: CUDA kernel with Array-of-Structures layout.
- **GPU SoA fused**: CUDA kernel with SoA layout, force+integration in one kernel, block size 256.
- **GPU SoA unfused**: SoA but two kernels (force compute, then integrate) for comparison.
- **GPU SoA auto-tune**: Same fused kernel, tries block sizes 128/256/512 and reports the best time.

A benchmark harness runs all five variants for multiple particle counts (1K to 400K) and writes timing results to CSV. A Python script plots the results.

## Build and run

Requirements: CMake 3.18+, C++17 compiler, CUDA toolkit.

```bash
mkdir build && cd build
cmake ..
cmake --build .
./gpu_particle_benchmark
```

Run the benchmark (no window, writes CSV):

```bash
./gpu_particle_benchmark --benchmark
```

Or run the visualizer:

```bash
./gpu_particle_visualizer
```

To plot benchmark results:

```bash
pip install matplotlib
python scripts/plot_results.py
```

The plot is saved as `benchmark_plot.png`.

## Visualizer (optional)

A 2D real-time visualizer shows particle positions (x,y), FPS, and current method.

**Requirements:** SDL2 and SDL2_ttf (e.g. `vcpkg install sdl2 sdl2-ttf`, or system packages on Linux).

If found at configure time, the `gpu_particle_visualizer` target is built. Run it from the build directory:

```bash
./gpu_particle_visualizer   # or .\Release\gpu_particle_visualizer.exe on Windows
```

- **FPS** and current **Method** are shown in the top-left.
- **Keys 1–5** switch approach: `1` CPU, `2` GPU Naive, `3` GPU SoA Fused (256), `4` GPU SoA Unfused, `5` GPU SoA Fused (512).
- **Escape** quits.

Particle count is fixed at 50,000 in the visualizer.

## Results

Sample benchmark output (100 steps, hardware-dependent):

| N      | CPU (ms) | GPU Naive (ms) | GPU SoA Fused (ms) | GPU SoA Unfused (ms) | GPU Auto-tune (ms) |
|--------|----------|----------------|--------------------|----------------------|--------------------|
| 1,000  | 0.75     | 1.05           | 1.72               | 3.67                 | 1.35               |
| 10,000 | 7.93     | 1.39           | 1.64               | 1.93                 | 0.91               |
| 50,000 | 38.83    | 2.59           | 1.22               | 2.59                 | 1.22               |
| 100,000| 86.15    | 3.91           | 1.53               | 3.20                 | 1.29               |
| 200,000| 205.43   | 6.73           | 5.43               | 9.51                 | 4.41               |
| 400,000| 396.76   | 14.99          | 11.09              | 20.28                | 9.64               |

GPU SoA fused and auto-tune typically outperform the CPU for larger N. Run `--benchmark` on your machine to get your own numbers.

## Why do particles look different for each approach?

When you switch between CPU, GPU Naive, GPU SoA Fused, etc., you may notice the particle patterns differ slightly even though all use the same physics. This is expected.

**Cause: floating-point non-associativity.** The same math (spring force, Euler integration) is evaluated in a different order:

- **CPU**: processes particles sequentially, one after another.
- **GPU**: processes many particles in parallel; execution order varies by thread/warp/block.
- **Layout**: AoS vs SoA changes memory access and often the sequence of arithmetic.

Floating-point addition and multiplication are not associative: `(a + b) + c` can differ slightly from `a + (b + c)` due to rounding. Those small differences accumulate over many frames, so particle positions diverge. The simulation is still physically correct; the trajectories are just numerically different.

## Why is FPS fluctuating?

FPS in the visualizer can vary due to VSync (capping to monitor refresh rate), GPU scheduling, thermal throttling, or other processes. For stable performance numbers, use the benchmark mode (`--benchmark`) and check the CSV output.

## Naive vs optimized kernels

| Variant       | Layout             | Kernels                       | Block size   |
|---------------|--------------------|-------------------------------|--------------|
| Naive (AoS)   | `struct Particle`  | 1                             | 256          |
| SoA fused     | Separate arrays    | 1 (force + integrate)         | 256          |
| SoA unfused   | Separate arrays    | 2 (force, then integrate)     | 256          |
| SoA auto-tune | Separate arrays    | 1 (fused)                     | Best of 128/256/512 |

SoA improves GPU performance because adjacent threads in a warp access adjacent addresses, enabling efficient memory coalescing.

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
  visualizer_launcher.cpp
scripts/
  plot_results.py
README.md
```
