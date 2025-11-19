#pragma once

#include "pattern_detection.cuh"
#include <vector>

/**
 * Real-time Turing pattern simulation for interactive viewing.
 *
 * Manages GPU-based Gray-Scott simulation at high resolution (1000x1000)
 */
class RealtimePatternSimulation {
private:
    double *d_u_ping, *d_v_ping;    // Buffer A
    double *d_u_pong, *d_v_pong;    // Buffer B
    int active_buffer = 0;         // Which buffer is "current"

    // Simulation parameters
    int Nx, Ny;                // Grid dimensions (e.g., 1000x1000)
    double dx;
    double dt;             // Spatial and temporal resolution
    ParamSet params;           // f, k, du, dv, dx, dt
    int current_timestep;      // Current simulation timestep

    // CUDA grid configuration
    dim3 threads;              // Threads per block (e.g., 16x16)
    dim3 blocks;               // Number of blocks

    // Helper methods
    void allocate_gpu_memory();
    void free_gpu_memory();
    void initialize_perturbation();
    double calculate_stable_dt(double dx, double du, double dv);

public:
    /**
     * Constructor: Initialize simulation with given parameters and grid size.
     *
     * @param p Gray-Scott parameters (f, k, du, dv)
     * @param grid_size Grid dimension (creates grid_size x grid_size grid)
     */
    RealtimePatternSimulation(ParamSet p, int grid_size = 1000);

    ~RealtimePatternSimulation();

    /**
     * Run N simulation timesteps using ping-pong buffers.
     *
     * @param num_steps Number of timesteps to execute
     */
    void step(int num_steps);

    /**
     * Copy current simulation state from GPU to CPU.
     *
     * @param u_out Output vector for U field (will be resized to Nx*Ny)
     * @param v_out Output vector for V field (will be resized to Nx*Ny)
     */
    void get_state(std::vector<double>& u_out, std::vector<double>& v_out);

    /**
     * Reset simulation to initial condition.
     */
    void reset();

    /**
     * Get current timestep number.
     */
    int get_timestep() const { return current_timestep; }

    /**
     * Get simulation parameters.
     */
    ParamSet get_params() const { return params; }

    /**
     * Get grid dimensions.
     */
    int get_nx() const { return Nx; }
    int get_ny() const { return Ny; }
};
