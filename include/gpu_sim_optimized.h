#pragma once

/// GPU optimized simulation using Structure-of-Arrays (SoA) layout.
/// All pointers must be device-allocated and have length n.
/// Uses tuned block size and coalesced memory access for better performance.
///
/// \param d_x, d_y, d_z      Device position arrays
/// \param d_vx, d_vy, d_vz   Device velocity arrays
/// \param n                  Number of particles
/// \param dt                 Time step (s)
/// \param steps              Number of integration steps
void simulate_gpu_optimized(float* d_x, float* d_y, float* d_z,
                           float* d_vx, float* d_vy, float* d_vz,
                           int n, float dt, int steps);

/// Same as simulate_gpu_optimized but with explicit block size (for auto-tune).
void simulate_gpu_optimized_blocksize(float* d_x, float* d_y, float* d_z,
                                     float* d_vx, float* d_vy, float* d_vz,
                                     int n, float dt, int steps, int blockSize);
