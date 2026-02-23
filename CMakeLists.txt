cmake_minimum_required(VERSION 3.18)
project(gpu-particle-benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(gpu_particle_benchmark
    src/main.cpp
    src/cpu_sim.cpp
    src/benchmark.cpp
    src/gpu_sim_naive.cu
    src/gpu_sim_optimized.cu
    src/gpu_sim_soa_unfused.cu
)

target_include_directories(gpu_particle_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(gpu_particle_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
# CUDA 13.x supports sm_75 (Turing) through Blackwell only.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    set_property(TARGET gpu_particle_benchmark PROPERTY CUDA_ARCHITECTURES 75 80 86 89 90)
endif()
