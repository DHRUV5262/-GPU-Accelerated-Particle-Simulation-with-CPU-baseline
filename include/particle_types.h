#pragma once

// Physical constants for the particle simulation.
// spring_k: spring constant for force pulling particles toward origin (thread-like)
constexpr float kGravityY = 0.0f;   // no gravity; spring alone pulls back to center
constexpr float kSpringK = 0.5f;

// Particle struct for Array-of-Structures (AoS) layout (GPU naive).
// Contains position (x,y,z) and velocity (vx,vy,vz).
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};
