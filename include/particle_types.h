#pragma once

// Physical constants for the particle simulation.
// gravity: downward acceleration (m/s^2)
// spring_k: spring constant for force pulling particles toward origin
constexpr float kGravityY = -9.81f;
constexpr float kSpringK = 0.5f;

// Particle struct for Array-of-Structures (AoS) layout (GPU naive).
// Contains position (x,y,z) and velocity (vx,vy,vz).
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};
