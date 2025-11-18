#include "../include/pattern_detection.cuh"
#include "../include/definitions.cuh"  // For gpuErrchk macro
#include <cmath>

// =============================================================================
// DEVICE HELPER FUNCTIONS
// =============================================================================

// Compute flat array index from 3D position (particle, x, y)
static __inline__ __device__ int idx_3d(int particle_id, int x, int y, int Nx, int Ny) 
{
    return particle_id * Nx * Ny + y * Nx + x;
}

// Compute 2D Laplacian with periodic boundaries for mega-batch
static __inline__ __device__ double laplace_batch(
    const double* data,
    int particle_id,
    int x,
    int y,
    int Nx,
    int Ny
) 
{
    int left = (x == 0) ? Nx - 1 : x - 1;
    int right = (x == Nx - 1) ? 0 : x + 1;
    int up = (y == 0) ? Ny - 1 : y - 1;
    int down = (y == Ny - 1) ? 0 : y + 1;

    return data[idx_3d(particle_id, left, y, Nx, Ny)] + 
        data[idx_3d(particle_id, right, y, Nx, Ny)] + 
        data[idx_3d(particle_id, x, up, Nx, Ny)] + 
        data[idx_3d(particle_id, x, down, Nx, Ny)] - 
        4.0 * data[idx_3d(particle_id, x, y, Nx, Ny)];
}

// =============================================================================
// KERNEL IMPLEMENTATIONS
// =============================================================================

__global__ void initialize_fields(
    double* u_data,
    double* v_data,
    int num_particles,
    int Nx,
    int Ny
) 
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int particle_id = blockIdx.z;

    if (x >= Nx || y >= Ny || particle_id >= num_particles) return;

    const int flat_idx = idx_3d(particle_id, x, y, Nx, Ny);

    // Initialize u = 1.0, v = 0.0
    u_data[flat_idx] = 1.0;
    v_data[flat_idx] = 0.0;

    if ((x <= Nx/2 + 2 && x >= Nx/2 - 2) && (y <= Ny/2 + 2 && y >= Ny/2 - 2))
        v_data[flat_idx] = 1.0;
}

__global__ void compute_time_step_batch(
    const double* u_in,
    const double* v_in,
    double* u_out,
    double* v_out,
    const ParamSet* params,
    int num_particles,
    int Nx,
    int Ny
) 
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int particle_id = blockIdx.z;

    if (x >= Nx || y >= Ny || particle_id >= num_particles) return;

    const int flat_idx = idx_3d(particle_id, x, y, Nx, Ny);

    // Get particle params
    ParamSet p = params[particle_id];

    // Get current u and v
    double u = u_in[flat_idx];
    double v = v_in[flat_idx];

    double laplace_u = laplace_batch(u_in, particle_id, x, y, Nx, Ny);
    double laplace_v = laplace_batch(v_in, particle_id, x, y, Nx, Ny);

    // Write new time step (using Gray-Scott model here)
    u_out[flat_idx] = u + p.dt * (p.du * laplace_u - u*v*v + p.f * (1-u));
    v_out[flat_idx] = v + p.dt * (p.dv * laplace_v + u*v*v - v * (p.f + p.k));
}

// Use root mean square to see if our pattern has significantly changed
__global__ void compute_temporal_difference(
    const double* u_current,
    const double* u_snapshot,
    double* rms_differences,
    int num_particles,
    int Nx,
    int Ny
) 
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int particle_id = blockIdx.z;

    if (x >= Nx || y >= Ny || particle_id >= num_particles) return;

    const int flat_idx = idx_3d(particle_id, x, y, Nx, Ny);

    double diff = u_current[flat_idx] - u_snapshot[flat_idx];
    double sq_diff = diff * diff;

    atomicAdd(&rms_differences[particle_id], sq_diff);

    // TODO (later, after kernel launch):
    // On host or in another kernel, divide by (Nx*Ny) and take sqrt
}

__global__ void compute_power_spectrum(
    const cufftDoubleComplex* freq_data,
    double* power_data,
    int total_elements
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total_elements) return;

    cufftDoubleComplex c = freq_data[idx];
    power_data[idx] = c.x * c.x + c.y * c.y;
}

__global__ void analyze_power_spectrum(
    const double* power_spectrum,
    PatternResult* results,
    int num_particles,
    int Nx,
    int Ny_half
) 
{
    int particle_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (particle_id >= num_particles) return;

    double max_power = 0.0;
    int max_kx = 0, max_ky = 0;
    double sum_power = 0.0;
    int count = 0;
    
    // Loop through all frequencies for this particle
    for (int ky = 0; ky < Ny_half; ky++) 
    {
        for (int kx = 0; kx < Nx; kx++) 
        {
            // Skip DC (kx==0 && ky==0)
            if (kx == 0 && ky == 0) continue;

            int idx = particle_id*Ny_half*Nx + ky*Nx + kx;
            double power = power_spectrum[idx];
            
            // Compute actual wave number (handle wrapping for kx > Nx/2)
            int actual_kx = (kx <= Nx/2) ? kx : kx - Nx;
            double k = sqrt(actual_kx*actual_kx + ky*ky);
            
            // Accumulate for average
            sum_power += power;
            count++;
            
            if (k >= MIN_WAVE_NUMBER && k <= MAX_WAVE_NUMBER)
            {
                if (power > max_power)
                {
                    max_power = power;
                    max_kx = kx;
                    max_ky = ky;
                }
            }
        }
    }
    
    double avg_power = sum_power / count;
    results[particle_id].spatial_ratio = max_power / avg_power;
    results[particle_id].peak_kx = max_kx;
    results[particle_id].peak_ky = max_ky;
    results[particle_id].peak_wave_number = sqrt(max_kx*max_kx + max_ky*max_ky);
}

// =============================================================================
// HOST FUNCTION: Main Pipeline
// =============================================================================

std::vector<PatternResult> detect_patterns_batch(
    const std::vector<ParamSet>& param_sets
) {
    const int num_particles = param_sets.size();

    const int Nx = PATTERN_NX;
    const int Ny = PATTERN_NY;
    const int grid_size = Nx * Ny;
    const int total_size = num_particles * grid_size;

    // Allocate device memory
    double* d_u_ping;
    double* d_v_ping;
    double* d_u_pong;
    double* d_v_pong;
    double* d_u_snapshot;
    double* d_v_snapshot;
    ParamSet* d_params;
    double* d_rms_diff;
    gpuErrchk(cudaMalloc(&d_u_ping, sizeof(double) * total_size));
    gpuErrchk(cudaMalloc(&d_v_ping, sizeof(double) * total_size));
    gpuErrchk(cudaMalloc(&d_u_pong, sizeof(double) * total_size));
    gpuErrchk(cudaMalloc(&d_v_pong, sizeof(double) * total_size));
    gpuErrchk(cudaMalloc(&d_u_snapshot, sizeof(double) * total_size));
    gpuErrchk(cudaMalloc(&d_v_snapshot, sizeof(double) * total_size));
    gpuErrchk(cudaMalloc(&d_params, sizeof(ParamSet) * num_particles));
    gpuErrchk(cudaMalloc(&d_rms_diff, sizeof(double) * num_particles));
    gpuErrchk(cudaMemset(d_rms_diff, 0, sizeof(double) * num_particles));

    // Copy parameters to device
    gpuErrchk(cudaMemcpy(
        d_params, 
        param_sets.data(), 
        num_particles * sizeof(ParamSet), 
        cudaMemcpyHostToDevice
    ));

    // Set up 3D grid/block dimensions for kernel launches
    dim3 threads(16, 16, 1);
    dim3 blocks((Nx+15)/16, (Ny+15)/16, num_particles);

    // =========================================================================
    // PHASE 1: Initialize fields
    // =========================================================================

    initialize_fields<<<blocks, threads>>>(d_u_ping, d_v_ping, num_particles, Nx, Ny);
    gpuErrchk(cudaDeviceSynchronize());

    // =========================================================================
    // PHASE 2: Run initial simulation (pattern formation)
    // =========================================================================

    for (int i = 0; i < INITIAL_TIMESTEPS; i++)
    {
        compute_time_step_batch<<<blocks, threads>>>(
            d_u_ping, 
            d_v_ping,
            d_u_pong,
            d_v_pong,
            d_params,
            num_particles,
            Nx,
            Ny
        );

        // Swap pointers
        std::swap(d_u_ping, d_u_pong);
        std::swap(d_v_ping, d_v_pong);
    }

    // =========================================================================
    // PHASE 3: Save snapshot
    // =========================================================================

    gpuErrchk(cudaMemcpy(
        d_u_snapshot,
        d_u_ping,
        sizeof(double) * total_size,
        cudaMemcpyDeviceToDevice
    ));

    gpuErrchk(cudaMemcpy(
        d_v_snapshot,
        d_v_ping,
        sizeof(double) * total_size,
        cudaMemcpyDeviceToDevice
    ));

    // =========================================================================
    // PHASE 4: Run extended simulation (stability check)
    // =========================================================================

    for (int i = 0; i < STABILITY_TIMESTEPS; i++)
    {
        compute_time_step_batch<<<blocks, threads>>>(
            d_u_ping, 
            d_v_ping,
            d_u_pong,
            d_v_pong,
            d_params,
            num_particles,
            Nx,
            Ny
        );

        // Swap pointers
        std::swap(d_u_ping, d_u_pong);
        std::swap(d_v_ping, d_v_pong);
    }

    // =========================================================================
    // PHASE 5: Compute temporal difference
    // =========================================================================

    compute_temporal_difference<<<blocks, threads>>>(
        d_u_ping,
        d_u_snapshot,
        d_rms_diff,
        num_particles,
        Nx,
        Ny
    );

    // Copy RMS differences to host and finalize
    std::vector<double> h_rms_diff(num_particles);
    gpuErrchk(cudaMemcpy(
        h_rms_diff.data(),
        d_rms_diff,
        sizeof(double) * num_particles,
        cudaMemcpyDeviceToHost
    ));

    // Finalize RMS: divide by grid_size and take sqrt
    for (int i = 0; i < num_particles; i++) {
        h_rms_diff[i] = sqrt(h_rms_diff[i] / grid_size);
    }

    // =========================================================================
    // PHASE 6: FFT analysis
    // =========================================================================

    // TODO: Allocate d_freq (complex output) and d_power (power spectrum)
    cufftDoubleComplex* d_freq;
    double* d_power;
    gpuErrchk(cudaMalloc(
        &d_freq, 
        sizeof(cufftDoubleComplex) * num_particles * Nx * (Ny/2 + 1)
    ));
    gpuErrchk(cudaMalloc(
        &d_power, 
        sizeof(double) * num_particles * Nx * (Ny/2 + 1)
    ));

    // TODO: Create batched FFT plan using cufftPlanMany
    cufftHandle plan;
    int rank = 2;
    int n[] = {Ny, Nx};

    // Input embedding
    int inembed[] = {Ny, Nx};
    int istride = 1;
    int idist = Nx * Ny;

    // Output embedding
    int onembed[] = {Ny, Nx/2 + 1};
    int ostride = 1;
    int odist = Nx * (Ny/2 + 1);

    int batch = num_particles;

    cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed,
        ostride, odist, CUFFT_D2Z,batch);

    // Execute batched FFT
    cufftExecD2Z(plan, d_u_ping, d_freq);

    // Compute power spectrum: |freq|^2
    // This is a 1D kernel processing all frequency elements
    int freq_size = num_particles * Nx * (Ny/2 + 1);
    int freq_threads = 256;
    int freq_blocks = (freq_size + freq_threads - 1) / freq_threads;
    compute_power_spectrum<<<freq_blocks, freq_threads>>>(d_freq, d_power, freq_size);


    // =========================================================================
    // PHASE 7: Analyze power spectrum
    // =========================================================================

    PatternResult* d_results;
    gpuErrchk(cudaMalloc(&d_results, sizeof(PatternResult) * num_particles));

    // Analyze power spectrum: one thread per particle
    int power_threads = 256;
    int power_blocks = (num_particles + power_threads - 1) / power_threads;
    analyze_power_spectrum<<<power_blocks, power_threads>>>(
        d_power, d_results, num_particles, Nx, (Ny/2 + 1));

    // =========================================================================
    // PHASE 8: Classify and copy results back
    // =========================================================================

    std::vector<PatternResult> results(num_particles);
    gpuErrchk(cudaMemcpy(results.data(), d_results, sizeof(PatternResult) * num_particles, 
        cudaMemcpyDeviceToHost));

    // Fill pattern structs
    for (int i = 0; i < num_particles; i++)
    {
        results[i].params = param_sets[i];
        results[i].temporal_change = h_rms_diff[i];
        if (results[i].spatial_ratio > SPATIAL_RATIO_THRESHOLD
            && results[i].temporal_change < TEMPORAL_CHANGE_THRESHOLD)
        {
            results[i].classification = PatternType::TURING_PATTERN;
        }
        else if (results[i].spatial_ratio > SPATIAL_RATIO_THRESHOLD)
        {
            results[i].classification = PatternType::OSCILLATING_PATTERN;
        }
        else
        {
            results[i].classification = PatternType::NO_PATTERN;
        }

        if (results[i].classification != PatternType::NO_PATTERN)
        {
            results[i].u_final.resize(grid_size);
            results[i].v_final.resize(grid_size);
            gpuErrchk(cudaMemcpy(
                results[i].u_final.data(),
                d_u_ping + i * grid_size,
                sizeof(double) * grid_size,
                cudaMemcpyDeviceToHost
            ));
            gpuErrchk(cudaMemcpy(
                results[i].v_final.data(),
                d_v_ping + i * grid_size,
                sizeof(double) * grid_size,
                cudaMemcpyDeviceToHost
            ));
        }
    }

    // Clean up device memory
    gpuErrchk(cudaFree(d_u_ping));
    gpuErrchk(cudaFree(d_v_ping));
    gpuErrchk(cudaFree(d_u_pong));
    gpuErrchk(cudaFree(d_v_pong));
    gpuErrchk(cudaFree(d_u_snapshot));
    gpuErrchk(cudaFree(d_v_snapshot));
    gpuErrchk(cudaFree(d_params));
    gpuErrchk(cudaFree(d_rms_diff));
    gpuErrchk(cudaFree(d_freq));
    gpuErrchk(cudaFree(d_power));
    gpuErrchk(cudaFree(d_results));
    cufftDestroy(plan);

    // Placeholder return
    return results;
}
