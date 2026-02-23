#pragma once

/// CPU baseline simulation using Structure-of-Arrays (SoA) layout.
/// Single-threaded, clear and readable reference implementation.
///
/// \param x, y, z     Position arrays (length n)
/// \param vx, vy, vz  Velocity arrays (length n)
/// \param n           Number of particles
/// \param dt          Time step (s)
/// \param steps       Number of integration steps
void simulate_cpu(float* x, float* y, float* z,
                 float* vx, float* vy, float* vz,
                 int n, float dt, int steps);
