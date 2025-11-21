#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include "tuning_parameters.h"

// Pattern classification
enum PatternType {
    NO_PATTERN,
    TURING_PATTERN,        // Spatially periodic + temporally stable
    OSCILLATING_PATTERN    // Spatially periodic but temporally unstable
};

// Input: Parameters for one simulation
struct ParamSet {
    double f;   // Feed rate
    double k;   // Kill rate
    double du;  // Diffusion rate for u
    double dv;  // Diffusion rate for v
    double dx;  // Spatial step
    double dt;  // Time step
};

// Output: Analysis results for one simulation
struct PatternResult {
    // Input parameters (copy for convenience)
    ParamSet params;

    // Spatial periodicity analysis (from FFT)
    double spatial_ratio;      // Peak power / average power
    double max_power;          // Absolute peak power
    double peak_wave_number;   // Magnitude of dominant frequency
    int peak_kx;               // X component of peak frequency
    int peak_ky;               // Y component of peak frequency

    // Temporal stability analysis
    double temporal_change;    // RMS difference between snapshots

    // Final classification
    PatternType classification;

    // Final state (only allocated if pattern found)
    std::vector<double> u_final;
    std::vector<double> v_final;
};

// =============================================================================
// CUDA KERNEL DECLARATIONS
// =============================================================================

// Initialize u/v fields with perturbation in center
__global__ void initialize_fields(
    double* u_data,
    double* v_data,
    int num_particles,
    int Nx,
    int Ny
);

// Mega-batch Gray-Scott time step (3D kernel)
__global__ void compute_time_step_batch(
    const double* u_in,
    const double* v_in,
    double* u_out,
    double* v_out,
    const ParamSet* params,
    int num_particles,
    int Nx,
    int Ny
);

// Compute RMS difference between current state and snapshot (temporal stability)
__global__ void compute_temporal_difference(
    const double* u_current,
    const double* u_snapshot,
    double* rms_differences,  // Output: one value per particle
    int num_particles,
    int Nx,
    int Ny
);

// Analyze power spectrum to find peak and compute spatial ratio
__global__ void analyze_power_spectrum(
    const double* power_spectrum,  // Input: power spectrum for all particles
    PatternResult* results,         // Output: write spatial analysis results
    int num_particles,
    int Nx,
    int Ny_half
);

// =============================================================================
// HOST FUNCTION DECLARATIONS
// =============================================================================

// Run the full pattern detection pipeline on a batch of parameter sets
std::vector<PatternResult> detect_patterns_batch(
    const std::vector<ParamSet>& param_sets
);
