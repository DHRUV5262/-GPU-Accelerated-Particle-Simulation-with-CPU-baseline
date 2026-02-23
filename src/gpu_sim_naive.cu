#include "gpu_sim_naive.h"
#include "particle_types.h"

#include <cuda_runtime.h>

__global__ void particle_step_naive_kernel(Particle* particles, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle& p = particles[i];

    // Force: spring toward origin + gravity
    float ax = -kSpringK * p.x;
    float ay = -kSpringK * p.y + kGravityY;
    float az = -kSpringK * p.z;

    // Semi-implicit Euler
    p.vx += ax * dt;
    p.vy += ay * dt;
    p.vz += az * dt;

    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.z += p.vz * dt;
}

void simulate_gpu_naive(Particle* d_particles, int n, float dt, int steps) {
    const int blockSize = 256;  // Common default; naive kernel doesn't tune this
    const int numBlocks = (n + blockSize - 1) / blockSize;

    for (int s = 0; s < steps; ++s) {
        particle_step_naive_kernel<<<numBlocks, blockSize>>>(d_particles, n, dt);
    }
    cudaDeviceSynchronize();
}
