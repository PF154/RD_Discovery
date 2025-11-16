cmake_minimum_required(VERSION 3.18)
project(ReactionDiffusion LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# CUDA architecture - adjust based on your GPU
# Common values: 50, 52, 60, 61, 70, 75, 80, 86, 89, 90
# Use 'Auto' to detect automatically or specify your GPU architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

# Find SFML (system-installed)
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Add executable
add_executable(reaction_diffusion
    reaction_diffusion.cu
    definitions.cuh
)

# Set properties for CUDA files
set_target_properties(reaction_diffusion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Include directories
target_include_directories(reaction_diffusion PRIVATE
    ${CMAKE_SOURCE_DIR}
)

# Link SFML and cuFFT libraries
target_link_libraries(reaction_diffusion PRIVATE
    sfml-graphics
    sfml-window
    sfml-system
    cufft
)

# CUDA specific compiler flags
target_compile_options(reaction_diffusion PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -use_fast_math
    >
)

# Print configuration info
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
