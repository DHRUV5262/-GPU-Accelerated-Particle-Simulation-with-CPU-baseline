#include "cpu_sim.h"
#include "particle_types.h"

#include <cmath>

void simulate_cpu(float* x, float* y, float* z,
                 float* vx, float* vy, float* vz,
                 int n, float dt, int steps) {
    for (int s = 0; s < steps; ++s) {
        for (int i = 0; i < n; ++i) {
            // Force: spring toward origin + gravity
            // F = -k * pos, gravity = (0, -9.81, 0)
            float ax = -kSpringK * x[i];
            float ay = -kSpringK * y[i] + kGravityY;
            float az = -kSpringK * z[i];

            // Semi-implicit Euler: v += a * dt, x += v * dt
            vx[i] += ax * dt;
            vy[i] += ay * dt;
            vz[i] += az * dt;

            x[i] += vx[i] * dt;
            y[i] += vy[i] * dt;
            z[i] += vz[i] * dt;
        }
    }
}
