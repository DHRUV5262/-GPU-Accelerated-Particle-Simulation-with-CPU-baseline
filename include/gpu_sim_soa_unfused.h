#pragma once

/// GPU SoA simulation with separate force and integration kernels (NOT fused).
/// Used to compare fused vs non-fused performance.
/// All pointers must be device-allocated and have length n.
void simulate_gpu_soa_unfused(float* d_x, float* d_y, float* d_z,
                              float* d_vx, float* d_vy, float* d_vz,
                              int n, float dt, int steps);
