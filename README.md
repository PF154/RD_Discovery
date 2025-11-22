# RD_Discovery

This is a project which I began under the supervision of John Whitman at SimBioSys.
The aim of the project is to use a particle swarm technique to discover parameter combinations that create Turing patterns in the Gray-Scott reaction-diffusion model.

Gray Scott model:

[Insert model here]

## Techniques Used

### Concurrency
This program employs a classic producer-consumer model to send data between the particle simulation thread, which handles the UI and visualization, and the pattern detection thread, which must compute massive amounts of data on the GPU for every frame. If the UI were to wait on the GPU to finish, it would slow the program down beyond the point of usability. Instead, a thread safe queue is used to send work to the GPU (consumer), and render completed work to the screen (consumer).

### CUDA Pogramming
This project has to compute an enormous number of timesteps to determine if a particle has located a turing pattern. To handle this, it leverages CUDA to compute the timesteps in parallel, using mega-batch processing to further increase parallelism. This allows for rapid computation of the individual reaction-diffusion simulations, each of which must run for 7500 timesteps on a 100x100 grid.

### Quad tree data structure
The particle swarm algorithm which aims to effiently discover turing pattern parameters is based on algorithms which operate on a continuous surface. In these, a particle would save a local and global best value, which would then influence it's navigation. However, our search condition here is boolean: either a set of parameters generates a turing pattern or it doesn't. To adapt the particle swarm behvaior to this scenario, the code exploits the quad tree data structure to determine which discovered patterns should have their own "pull" on the particles, which should be clustered into a single point of influence (for those farther away from the particle), and which can be disregarded (too far away or otherwise non-influential).

## Dependencies

### System-Installed
- **SFML 2.5+** - Graphics library for windowing and rendering
  ```bash
  sudo apt install libsfml-dev
  ```
- **CUDA Toolkit** - For GPU acceleration
- **OpenGL** - Required by ImGui-SFML (usually pre-installed on Linux)

### External Libraries (Source Files)
The following libraries should be downloaded as source files into the `external/` directory:

#### ImGui (v1.91.5)
Download these files to `external/imgui/`:
- imgui.h
- imgui.cpp
- imgui_draw.cpp
- imgui_widgets.cpp
- imgui_tables.cpp
- imconfig.h
- imgui_internal.h
- imstb_rectpack.h
- imstb_textedit.h
- imstb_truetype.h

#### ImGui-SFML (v2.6.x)
Download these files to `external/imgui-sfml/`:
- imgui-SFML.h
- imgui-SFML.cpp
- imgui-SFML_export.h

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running

```bash
./build/reaction_diffusion
```