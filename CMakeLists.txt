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

# Find OpenGL (required by ImGui-SFML)
find_package(OpenGL REQUIRED)

# Add executable with ImGui source files
add_executable(reaction_diffusion
    # Main application
    src/main.cu
    # Project source files
    src/particle.cpp
    src/rendering.cpp
    src/pattern_detection.cu
    src/async_pattern_detector.cpp
    src/realtime_simulation.cu
    src/realtime_mode.cpp
    src/particle_swarm_mode.cpp
    src/utilities.cpp
    src/debug_tools.cpp
    src/pattern_quadtree.cpp
    # Project headers
    include/particle.h
    include/rendering.h
    include/definitions.cuh
    include/pattern_detection.cuh
    include/thread_safe_queue.h
    include/async_pattern_detector.h
    include/realtime_simulation.cuh
    include/pattern_viewer_state.h
    include/realtime_mode.h
    include/particle_swarm_mode.h
    include/sim_types.h
    include/utilities.h
    include/tuning_parameters.h
    include/debug_tools.h
    include/pattern_quadtree.h
    # ImGui core files
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/imgui_tables.cpp
    # ImGui-SFML binding
    external/imgui-sfml/imgui-SFML.cpp
)

# Set properties for CUDA files
set_target_properties(reaction_diffusion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Include directories
target_include_directories(reaction_diffusion PRIVATE
    ${CMAKE_SOURCE_DIR}/include              # Project headers
    ${CMAKE_SOURCE_DIR}/external/imgui       # ImGui headers
    ${CMAKE_SOURCE_DIR}/external/imgui-sfml  # ImGui-SFML headers
)

# Link SFML, OpenGL, and cuFFT libraries
target_link_libraries(reaction_diffusion PRIVATE
    sfml-graphics
    sfml-window
    sfml-system
    OpenGL::GL
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
