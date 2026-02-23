#pragma once

#include "particle_types.h"

/// GPU naive simulation using Array-of-Structures (AoS) layout.
/// d_particles must be device-allocated and contain n Particle structs.
///
/// \param d_particles  Device pointer to Particle array (AoS)
/// \param n            Number of particles
/// \param dt           Time step (s)
/// \param steps        Number of integration steps
void simulate_gpu_naive(Particle* d_particles, int n, float dt, int steps);
