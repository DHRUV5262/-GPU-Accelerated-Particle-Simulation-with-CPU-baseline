#include "gpu_sim_soa_unfused.h"
#include "particle_types.h"

#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

// Kernel 1: Compute acceleration from position (force only).
__global__ void compute_accel_kernel(
    const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
    float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    ax[i] = -kSpringK * x[i];
    ay[i] = -kSpringK * y[i] + kGravityY;
    az[i] = -kSpringK * z[i];
}

// Kernel 2: Integrate velocity and position using acceleration.
__global__ void integrate_kernel(
    float* __restrict__ x, float* __restrict__ y, float* __restrict__ z,
    float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
    const float* __restrict__ ax, const float* __restrict__ ay, const float* __restrict__ az,
    int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    vx[i] += ax[i] * dt;
    vy[i] += ay[i] * dt;
    vz[i] += az[i] * dt;

    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

void simulate_gpu_soa_unfused(float* d_x, float* d_y, float* d_z,
                              float* d_vx, float* d_vy, float* d_vz,
                              int n, float dt, int steps) {
    float* d_ax = nullptr;
    float* d_ay = nullptr;
    float* d_az = nullptr;
    cudaMalloc(&d_ax, n * sizeof(float));
    cudaMalloc(&d_ay, n * sizeof(float));
    cudaMalloc(&d_az, n * sizeof(float));

    const int numBlocks = (n + kBlockSize - 1) / kBlockSize;

    for (int s = 0; s < steps; ++s) {
        compute_accel_kernel<<<numBlocks, kBlockSize>>>(d_x, d_y, d_z, d_ax, d_ay, d_az, n);
        integrate_kernel<<<numBlocks, kBlockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                                    d_ax, d_ay, d_az, n, dt);
    }
    cudaDeviceSynchronize();

    cudaFree(d_ax);
    cudaFree(d_ay);
    cudaFree(d_az);
}
