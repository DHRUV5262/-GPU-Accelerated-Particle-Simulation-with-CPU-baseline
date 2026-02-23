#include "benchmark.h"
#include "cpu_sim.h"
#include "gpu_sim_naive.h"
#include "gpu_sim_optimized.h"
#include "gpu_sim_soa_unfused.h"
#include "particle_types.h"

#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " " << cudaGetErrorString(err) << std::endl;          \
        }                                                                      \
    } while (0)

#define CHECK_CUDA_LAST() CHECK_CUDA(cudaGetLastError())

namespace {

constexpr unsigned kRandomSeed = 42;

void init_particles(float* x, float* y, float* z,
                   float* vx, float* vy, float* vz,
                   int n) {
    std::mt19937 rng(kRandomSeed);
    std::uniform_real_distribution<float> pos(-5.0f, 5.0f);
    std::uniform_real_distribution<float> vel(-1.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
        x[i] = pos(rng);
        y[i] = pos(rng);
        z[i] = pos(rng);
        vx[i] = vel(rng);
        vy[i] = vel(rng);
        vz[i] = vel(rng);
    }
}

void soa_to_aos(const float* x, const float* y, const float* z,
                const float* vx, const float* vy, const float* vz,
                int n, Particle* particles) {
    for (int i = 0; i < n; ++i) {
        particles[i].x = x[i];
        particles[i].y = y[i];
        particles[i].z = z[i];
        particles[i].vx = vx[i];
        particles[i].vy = vy[i];
        particles[i].vz = vz[i];
    }
}

}  // namespace

void run_benchmarks(const std::vector<int>& particle_counts, int steps,
                   const std::string& csv_path) {
    std::ofstream out(csv_path);
    if (!out) {
        std::cerr << "Failed to open " << csv_path << " for writing." << std::endl;
        return;
    }
    std::string abs_path = std::filesystem::absolute(csv_path).string();
    std::cout << "Writing CSV to: " << abs_path << std::endl;

    // Check GPU is available before running
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable GPU found." << std::endl;
        return;
    }
    CHECK_CUDA(cudaSetDevice(0));

    out << "N,cpu_ms,gpu_naive_ms,gpu_opt_ms,gpu_soa_unfused_ms,gpu_autotune_ms\n";

    for (int n : particle_counts) {
        std::vector<float> x(n), y(n), z(n), vx(n), vy(n), vz(n);
        init_particles(x.data(), y.data(), z.data(),
                      vx.data(), vy.data(), vz.data(), n);

        // --- CPU timing (std::chrono) ---
        std::vector<float> x_cpu = x, y_cpu = y, z_cpu = z;
        std::vector<float> vx_cpu = vx, vy_cpu = vy, vz_cpu = vz;

        auto t0 = std::chrono::high_resolution_clock::now();
        simulate_cpu(x_cpu.data(), y_cpu.data(), z_cpu.data(),
                    vx_cpu.data(), vy_cpu.data(), vz_cpu.data(),
                    n, 0.016f, steps);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // --- GPU naive (AoS, CUDA events) ---
        std::vector<Particle> particles(n);
        soa_to_aos(x.data(), y.data(), z.data(),
                   vx.data(), vy.data(), vz.data(), n, particles.data());

        Particle* d_particles = nullptr;
        CHECK_CUDA(cudaMalloc(&d_particles, n * sizeof(Particle)));
        CHECK_CUDA(cudaMemcpy(d_particles, particles.data(), n * sizeof(Particle), cudaMemcpyHostToDevice));

        cudaEvent_t start_naive, stop_naive;
        CHECK_CUDA(cudaEventCreate(&start_naive));
        CHECK_CUDA(cudaEventCreate(&stop_naive));

        CHECK_CUDA(cudaEventRecord(start_naive));
        simulate_gpu_naive(d_particles, n, 0.016f, steps);
        CHECK_CUDA_LAST();  // Catches kernel launch / device sync errors
        CHECK_CUDA(cudaEventRecord(stop_naive));
        CHECK_CUDA(cudaEventSynchronize(stop_naive));

        float gpu_naive_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_naive_ms, start_naive, stop_naive));

        cudaEventDestroy(start_naive);
        cudaEventDestroy(stop_naive);
        cudaFree(d_particles);

        // --- GPU optimized (SoA, CUDA events) ---
        float* d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
        float* d_vx = nullptr, *d_vy = nullptr, *d_vz = nullptr;
        CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_z, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_vx, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_vy, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_vz, n * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_y, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_z, z.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_vx, vx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_vy, vy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_vz, vz.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        cudaEvent_t start_opt, stop_opt;
        CHECK_CUDA(cudaEventCreate(&start_opt));
        CHECK_CUDA(cudaEventCreate(&stop_opt));

        CHECK_CUDA(cudaEventRecord(start_opt));
        simulate_gpu_optimized(d_x, d_y, d_z, d_vx, d_vy, d_vz, n, 0.016f, steps);
        CHECK_CUDA_LAST();  // Catches kernel launch / device sync errors
        CHECK_CUDA(cudaEventRecord(stop_opt));
        CHECK_CUDA(cudaEventSynchronize(stop_opt));

        float gpu_opt_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_opt_ms, start_opt, stop_opt));

        cudaEventDestroy(start_opt);
        cudaEventDestroy(stop_opt);

        // --- GPU SoA unfused (two kernels: force + integrate) ---
        CHECK_CUDA(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_y, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_z, z.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_vx, vx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_vy, vy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_vz, vz.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        cudaEvent_t start_unfused, stop_unfused;
        CHECK_CUDA(cudaEventCreate(&start_unfused));
        CHECK_CUDA(cudaEventCreate(&stop_unfused));
        CHECK_CUDA(cudaEventRecord(start_unfused));
        simulate_gpu_soa_unfused(d_x, d_y, d_z, d_vx, d_vy, d_vz, n, 0.016f, steps);
        CHECK_CUDA_LAST();
        CHECK_CUDA(cudaEventRecord(stop_unfused));
        CHECK_CUDA(cudaEventSynchronize(stop_unfused));
        float gpu_unfused_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_unfused_ms, start_unfused, stop_unfused));
        cudaEventDestroy(start_unfused);
        cudaEventDestroy(stop_unfused);

        // --- GPU auto-tune: try block sizes 128, 256, 512, report best ---
        float gpu_autotune_ms = 1e9f;
        for (int blockSize : {128, 256, 512}) {
            CHECK_CUDA(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_y, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_z, z.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_vx, vx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_vy, vy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_vz, vz.data(), n * sizeof(float), cudaMemcpyHostToDevice));

            cudaEvent_t start_at, stop_at;
            CHECK_CUDA(cudaEventCreate(&start_at));
            CHECK_CUDA(cudaEventCreate(&stop_at));
            CHECK_CUDA(cudaEventRecord(start_at));
            simulate_gpu_optimized_blocksize(d_x, d_y, d_z, d_vx, d_vy, d_vz, n, 0.016f, steps, blockSize);
            CHECK_CUDA_LAST();
            CHECK_CUDA(cudaEventRecord(stop_at));
            CHECK_CUDA(cudaEventSynchronize(stop_at));
            float t = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&t, start_at, stop_at));
            cudaEventDestroy(start_at);
            cudaEventDestroy(stop_at);
            if (t < gpu_autotune_ms) gpu_autotune_ms = t;
        }

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        cudaFree(d_vx);
        cudaFree(d_vy);
        cudaFree(d_vz);

        out << n << "," << cpu_ms << "," << gpu_naive_ms << "," << gpu_opt_ms
            << "," << gpu_unfused_ms << "," << gpu_autotune_ms << "\n";
    }
}
