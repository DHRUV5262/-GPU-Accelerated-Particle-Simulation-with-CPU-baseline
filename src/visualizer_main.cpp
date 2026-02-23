/**
 * 2D particle visualizer: runs one sim approach per frame, draws x,y and FPS.
 * Keys 1-5: switch approach. Escape: quit.
 * Pass --benchmark (or -b) to run benchmark for different N and write CSV instead of opening the window.
 */
#include "benchmark.h"
#include "cpu_sim.h"
#include "gpu_sim_naive.h"
#include "gpu_sim_optimized.h"
#include "gpu_sim_soa_unfused.h"
#include "particle_types.h"

#include <SDL.h>
#include <cstring>
#include <iostream>
#include <SDL_ttf.h>
#include <cuda_runtime.h>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int WINDOW_W = 1280;
constexpr int WINDOW_H = 720;
constexpr float SIM_BOUND = 5.0f;   // particles in [-SIM_BOUND, SIM_BOUND]
constexpr float DT = 0.016f;
constexpr int STEPS_PER_FRAME = 1;
constexpr unsigned RAND_SEED = 42;

enum class Approach {
    CPU,
    GPU_NAIVE,
    GPU_SOA_FUSED,
    GPU_SOA_UNFUSED,
    GPU_SOA_512,
    COUNT
};

const char* approachName(Approach a) {
    switch (a) {
        case Approach::CPU: return "CPU";
        case Approach::GPU_NAIVE: return "GPU Naive (AoS)";
        case Approach::GPU_SOA_FUSED: return "GPU SoA Fused (256)";
        case Approach::GPU_SOA_UNFUSED: return "GPU SoA Unfused";
        case Approach::GPU_SOA_512: return "GPU SoA Fused (512)";
        default: return "?";
    }
}

// Start at center, random directions; spring pulls each back like a thread to center
void initParticles(std::vector<float>& x, std::vector<float>& y, std::vector<float>& z,
                  std::vector<float>& vx, std::vector<float>& vy, std::vector<float>& vz) {
    std::mt19937 rng(RAND_SEED);
    const float centerSpread = 0.2f;  // tiny cloud at center
    std::uniform_real_distribution<float> pos(-centerSpread, centerSpread);
    std::uniform_real_distribution<float> vel(-1.2f, 1.2f);  // random outward direction
    const int n = static_cast<int>(x.size());
    for (int i = 0; i < n; ++i) {
        x[i] = pos(rng);
        y[i] = pos(rng);
        z[i] = pos(rng);
        vx[i] = vel(rng);
        vy[i] = vel(rng);
        vz[i] = vel(rng);
    }
}

// Scale world x,y to screen (y flipped for screen coords)
void worldToScreen(float x, float y, int& sx, int& sy) {
    float u = (x + SIM_BOUND) / (2.0f * SIM_BOUND);
    float v = (y + SIM_BOUND) / (2.0f * SIM_BOUND);
    sx = static_cast<int>(u * (WINDOW_W - 1));
    sy = static_cast<int>((1.0f - v) * (WINDOW_H - 1));
    if (sx < 0) sx = 0;
    if (sx >= WINDOW_W) sx = WINDOW_W - 1;
    if (sy < 0) sy = 0;
    if (sy >= WINDOW_H) sy = WINDOW_H - 1;
}

} // namespace

static bool wantBenchmarkMode(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--benchmark") == 0 || std::strcmp(argv[i], "-b") == 0)
            return true;
    }
    return false;
}

int run_visualizer(int argc, char* argv[]) {
    // Toggle: if --benchmark / -b, run sim for different N and write CSV (no window)
    if (wantBenchmarkMode(argc, argv)) {
        std::vector<int> particle_counts = {1'000, 10'000, 50'000, 100'000};
        const int steps = 100;
        const std::string csv_path = "benchmark_results.csv";
        std::cout << "Running benchmarks (N = ";
        for (size_t i = 0; i < particle_counts.size(); ++i) {
            std::cout << particle_counts[i];
            if (i + 1 < particle_counts.size()) std::cout << ", ";
        }
        std::cout << ", steps = " << steps << ")...\n";
        run_benchmarks(particle_counts, steps, csv_path);
        std::cout << "Results written to " << csv_path << "\n";
        std::cout << "Run scripts/plot_results.py to generate the chart.\n";
        return 0;
    }

    const int N = 50000;
    if (SDL_Init(SDL_INIT_VIDEO) != 0) return 1;
    if (TTF_Init() != 0) return 1;

    SDL_Window* win = SDL_CreateWindow("Particle Visualizer - 1:CPU 2:Naive 3:SoA 4:Unfused 5:SoA512 Esc:Quit",
                                       SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                       WINDOW_W, WINDOW_H, 0);
    if (!win) return 1;
    SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    if (!ren) return 1;

    TTF_Font* font = nullptr;
#ifdef _WIN32
    const char* fontPaths[] = { "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/calibri.ttf" };
#else
    const char* fontPaths[] = { "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf", nullptr };
#endif
    for (int i = 0; fontPaths[i] != nullptr && !font; ++i)
        font = TTF_OpenFont(fontPaths[i], 24);

    // Host data (used for CPU sim and for drawing)
    std::vector<float> h_x(N), h_y(N), h_z(N), h_vx(N), h_vy(N), h_vz(N);
    initParticles(h_x, h_y, h_z, h_vx, h_vy, h_vz);

    // GPU resources (allocated once)
    Particle* d_particles = nullptr;
    float* d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    float* d_vx = nullptr, *d_vy = nullptr, *d_vz = nullptr;
    cudaMalloc(&d_particles, N * sizeof(Particle));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float));
    cudaMalloc(&d_vx, N * sizeof(float));
    cudaMalloc(&d_vy, N * sizeof(float));
    cudaMalloc(&d_vz, N * sizeof(float));

    auto syncHostToDevice = [&]() {
        cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_z, h_z.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vx, h_vx.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vy, h_vy.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vz, h_vz.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        std::vector<Particle> particles(N);
        for (int i = 0; i < N; ++i) {
            particles[i].x = h_x[i]; particles[i].y = h_y[i]; particles[i].z = h_z[i];
            particles[i].vx = h_vx[i]; particles[i].vy = h_vy[i]; particles[i].vz = h_vz[i];
        }
        cudaMemcpy(d_particles, particles.data(), N * sizeof(Particle), cudaMemcpyHostToDevice);
    };
    auto syncDeviceToHost = [&]() {
        cudaMemcpy(h_x.data(), d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_z.data(), d_z, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vx.data(), d_vx, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vy.data(), d_vy, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vz.data(), d_vz, N * sizeof(float), cudaMemcpyDeviceToHost);
    };
    syncHostToDevice();

    Approach current = Approach::GPU_SOA_FUSED;
    uint64_t freq = SDL_GetPerformanceFrequency();
    uint64_t lastTime = SDL_GetPerformanceCounter();
    float fps = 0.0f;

    bool quit = false;
    while (!quit) {
        uint64_t frameStart = SDL_GetPerformanceCounter();

        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = true;
            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) quit = true;
                if (e.key.keysym.sym == SDLK_1) {
                    if (current != Approach::CPU) syncDeviceToHost();
                    current = Approach::CPU;
                }
                if (e.key.keysym.sym == SDLK_2) { syncHostToDevice(); current = Approach::GPU_NAIVE; }
                if (e.key.keysym.sym == SDLK_3) { syncHostToDevice(); current = Approach::GPU_SOA_FUSED; }
                if (e.key.keysym.sym == SDLK_4) { syncHostToDevice(); current = Approach::GPU_SOA_UNFUSED; }
                if (e.key.keysym.sym == SDLK_5) { syncHostToDevice(); current = Approach::GPU_SOA_512; }
            }
        }

        // Run sim step(s)
        switch (current) {
            case Approach::CPU:
                simulate_cpu(h_x.data(), h_y.data(), h_z.data(),
                            h_vx.data(), h_vy.data(), h_vz.data(), N, DT, STEPS_PER_FRAME);
                break;
            case Approach::GPU_NAIVE:
                simulate_gpu_naive(d_particles, N, DT, STEPS_PER_FRAME);
                break;
            case Approach::GPU_SOA_FUSED:
                simulate_gpu_optimized_blocksize(d_x, d_y, d_z, d_vx, d_vy, d_vz, N, DT, STEPS_PER_FRAME, 256);
                break;
            case Approach::GPU_SOA_UNFUSED:
                simulate_gpu_soa_unfused(d_x, d_y, d_z, d_vx, d_vy, d_vz, N, DT, STEPS_PER_FRAME);
                break;
            case Approach::GPU_SOA_512:
                simulate_gpu_optimized_blocksize(d_x, d_y, d_z, d_vx, d_vy, d_vz, N, DT, STEPS_PER_FRAME, 512);
                break;
            default:
                break;
        }

        // Get positions for drawing (GPU: copy back)
        if (current == Approach::CPU) {
            // already in h_x, h_y
        } else if (current == Approach::GPU_NAIVE) {
            std::vector<Particle> particles(N);
            cudaMemcpy(particles.data(), d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
            for (int i = 0; i < N; ++i) {
                h_x[i] = particles[i].x;
                h_y[i] = particles[i].y;
            }
        } else {
            cudaMemcpy(h_x.data(), d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
        }

        // Draw
        SDL_SetRenderDrawColor(ren, 20, 20, 30, 255);
        SDL_RenderClear(ren);
        SDL_SetRenderDrawColor(ren, 180, 220, 255, 255);
        std::vector<SDL_Point> points(N);
        for (int i = 0; i < N; ++i) {
            worldToScreen(h_x[i], h_y[i], points[i].x, points[i].y);
        }
        SDL_RenderDrawPoints(ren, points.data(), N);

        // FPS text
        uint64_t now = SDL_GetPerformanceCounter();
        float frameMs = 1000.0f * static_cast<float>(now - lastTime) / static_cast<float>(freq);
        lastTime = now;
        fps = (frameMs > 0.0f) ? (1000.0f / frameMs) : 0.0f;

        if (font) {
            SDL_Color white = {255, 255, 255, 255};
            std::string fpsStr = "FPS: " + std::to_string(static_cast<int>(fps + 0.5f));
            SDL_Surface* surf = TTF_RenderText_Solid(font, fpsStr.c_str(), white);
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);
                if (tex) {
                    SDL_Rect rect = {10, 10, surf->w, surf->h};
                    SDL_RenderCopy(ren, tex, nullptr, &rect);
                    SDL_DestroyTexture(tex);
                }
                SDL_FreeSurface(surf);
            }
            std::string methodStr = "Method: ";
            methodStr += approachName(current);
            surf = TTF_RenderText_Solid(font, methodStr.c_str(), white);
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);
                if (tex) {
                    SDL_Rect rect = {10, 40, surf->w, surf->h};
                    SDL_RenderCopy(ren, tex, nullptr, &rect);
                    SDL_DestroyTexture(tex);
                }
                SDL_FreeSurface(surf);
            }
        }

        SDL_RenderPresent(ren);
    }

    if (font) TTF_CloseFont(font);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    TTF_Quit();
    SDL_Quit();

    cudaFree(d_particles);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);

    return 0;
}
