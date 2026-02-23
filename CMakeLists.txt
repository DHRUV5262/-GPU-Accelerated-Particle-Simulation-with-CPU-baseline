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

# --- Visualizer (optional: requires SDL2 and SDL2_ttf) ---
find_package(SDL2 QUIET)
find_package(SDL2_ttf QUIET)
if(SDL2_FOUND AND (SDL2_ttf_FOUND OR SDL2_TTF_FOUND))
    # C++ static lib (run_visualizer + cpu_sim + benchmark for --benchmark mode)
    add_library(visualizer_cpp STATIC src/visualizer_main.cpp src/cpu_sim.cpp src/benchmark.cpp)
    target_include_directories(visualizer_cpp PRIVATE ${CMAKE_SOURCE_DIR}/include)
    if(TARGET SDL2::SDL2 AND TARGET SDL2_ttf::SDL2_ttf)
        target_link_libraries(visualizer_cpp PRIVATE SDL2::SDL2 SDL2_ttf::SDL2_ttf)
    else()
        target_include_directories(visualizer_cpp PRIVATE ${SDL2_INCLUDE_DIRS} ${SDL2_TTF_INCLUDE_DIRS})
    endif()
    # CUDA static lib; separable OFF so no __cudaRegisterLinkedBinary when linking into exe
    add_library(visualizer_cuda STATIC
        src/gpu_sim_naive.cu
        src/gpu_sim_optimized.cu
        src/gpu_sim_soa_unfused.cu
    )
    target_include_directories(visualizer_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(visualizer_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
        set_property(TARGET visualizer_cuda PROPERTY CUDA_ARCHITECTURES 75 80 86 89 90)
    endif()
    # Exe: one .cpp that defines main() and calls run_visualizer; links both static libs
    add_executable(gpu_particle_visualizer src/visualizer_launcher.cpp)
    target_include_directories(gpu_particle_visualizer PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(gpu_particle_visualizer PRIVATE visualizer_cpp visualizer_cuda ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES})
    if(TARGET SDL2::SDL2 AND TARGET SDL2_ttf::SDL2_ttf)
        target_link_libraries(gpu_particle_visualizer PRIVATE SDL2::SDL2 SDL2_ttf::SDL2_ttf)
    else()
        target_include_directories(gpu_particle_visualizer PRIVATE
            ${SDL2_INCLUDE_DIRS} ${SDL2_TTF_INCLUDE_DIRS})
        target_link_libraries(gpu_particle_visualizer PRIVATE
            ${SDL2_LIBRARIES} ${SDL2_TTF_LIBRARIES})
    endif()
    message(STATUS "Visualizer enabled (SDL2 + SDL2_ttf found)")
else()
    message(STATUS "Visualizer skipped (install SDL2 and SDL2_ttf to build)")
endif()
