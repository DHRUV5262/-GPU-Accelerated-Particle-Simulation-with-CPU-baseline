#include "gpu_sim_optimized.h"
#include "particle_types.h"

#include <cuda_runtime.h>

// Block size 256: balances occupancy and register pressure on most GPUs.
// Warp size is 32, so 256 = 8 warps per block; good utilization without
// excessive register spilling. Easy to port to HIP (ROCm) where warp = 64.
constexpr int kBlockSize = 256;

__global__ void particle_step_optimized_kernel(
    float* __restrict__ x, float* __restrict__ y, float* __restrict__ z,
    float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
    int n, float dt) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // SoA: consecutive threads access consecutive memory → coalesced loads/stores.
    float xi = x[i], yi = y[i], zi = z[i];
    float vxi = vx[i], vyi = vy[i], vzi = vz[i];

    // Force: spring toward origin + gravity (single kernel for force + integration)
    float ax = -kSpringK * xi;
    float ay = -kSpringK * yi + kGravityY;
    float az = -kSpringK * zi;

    // Semi-implicit Euler
    vxi += ax * dt;
    vyi += ay * dt;
    vzi += az * dt;

    xi += vxi * dt;
    yi += vyi * dt;
    zi += vzi * dt;

    x[i] = xi;
    y[i] = yi;
    z[i] = zi;
    vx[i] = vxi;
    vy[i] = vyi;
    vz[i] = vzi;
}

void simulate_gpu_optimized(float* d_x, float* d_y, float* d_z,
                           float* d_vx, float* d_vy, float* d_vz,
                           int n, float dt, int steps) {
    simulate_gpu_optimized_blocksize(d_x, d_y, d_z, d_vx, d_vy, d_vz, n, dt, steps, kBlockSize);
}

void simulate_gpu_optimized_blocksize(float* d_x, float* d_y, float* d_z,
                                     float* d_vx, float* d_vy, float* d_vz,
                                     int n, float dt, int steps, int blockSize) {
    const int numBlocks = (n + blockSize - 1) / blockSize;

    for (int s = 0; s < steps; ++s) {
        particle_step_optimized_kernel<<<numBlocks, blockSize>>>(
            d_x, d_y, d_z, d_vx, d_vy, d_vz, n, dt);
    }
    cudaDeviceSynchronize();
}
