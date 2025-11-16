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

# Add SFML directory
set(SFML_DIR "${CMAKE_SOURCE_DIR}/SFML-3.0.0/lib/cmake/SFML")

# Find SFML
find_package(SFML 3.0 COMPONENTS Graphics Window System REQUIRED)

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
    ${CMAKE_SOURCE_DIR}/SFML-3.0.0/include
)

# Link SFML libraries
target_link_libraries(reaction_diffusion PRIVATE
    SFML::Graphics
    SFML::Window
    SFML::System
)

# CUDA specific compiler flags
target_compile_options(reaction_diffusion PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -use_fast_math
    >
)

# Set RPATH for SFML shared libraries
set_target_properties(reaction_diffusion PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/SFML-3.0.0/lib"
    INSTALL_RPATH "${CMAKE_SOURCE_DIR}/SFML-3.0.0/lib"
)

# Print configuration info
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "SFML Directory: ${SFML_DIR}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
