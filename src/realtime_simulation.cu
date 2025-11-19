#include "realtime_simulation.cuh"
#include "definitions.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>

static __inline__ __device__ double laplace_2d(
    const double* data,
    int x, int y,
    int Nx, int Ny
) {
    int left = (x == 0) ? Nx - 1 : x - 1;
    int right = (x == Nx - 1) ? 0 : x + 1;
    int up = (y == 0) ? Ny - 1 : y - 1;
    int down = (y == Ny - 1) ? 0 : y + 1;
    
    int idx = y * Nx + x;
    return data[y * Nx + left] + 
           data[y * Nx + right] + 
           data[up * Nx + x] + 
           data[down * Nx + x] - 
           4.0 * data[idx];
}

__global__ void compute_time_step(
    const double* u_in,
    const double* v_in,
    double* u_out,
    double* v_out,
    int Nx, int Ny,
    double f, double k,
    double du, double dv,
    double dx, double dt
) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= Nx || y >= Ny) return;
    
    int idx = y * Nx + x;
    double u = u_in[idx];
    double v = v_in[idx];
    
    double laplace_u = laplace_2d(u_in, x, y, Nx, Ny);
    double laplace_v = laplace_2d(v_in, x, y, Nx, Ny);
    
    u_out[idx] = u + dt * (du * laplace_u - u*v*v + f * (1-u));
    v_out[idx] = v + dt * (dv * laplace_v + u*v*v - v * (f + k));
}

RealtimePatternSimulation::RealtimePatternSimulation(ParamSet p, int grid_size)
    : params(p), Nx(grid_size), Ny(grid_size), current_timestep(0)
{
    dx = p.dx;
    dt = p.dt;

    threads = dim3(16, 16);
    blocks = dim3((Nx + 15) / 16, (Ny + 15) / 16); 

    allocate_gpu_memory();

    initialize_perturbation();
}

RealtimePatternSimulation::~RealtimePatternSimulation()
{
    free_gpu_memory();
}

void RealtimePatternSimulation::allocate_gpu_memory()
{
    const int total_size = Nx * Ny * sizeof(double);

    gpuErrchk(cudaMalloc(&d_u_ping, total_size));
    gpuErrchk(cudaMalloc(&d_v_ping, total_size));
    gpuErrchk(cudaMalloc(&d_u_pong, total_size));
    gpuErrchk(cudaMalloc(&d_v_pong, total_size));
}

void RealtimePatternSimulation::free_gpu_memory()
{
    gpuErrchk(cudaFree(d_u_ping));
    gpuErrchk(cudaFree(d_v_ping));
    gpuErrchk(cudaFree(d_u_pong));
    gpuErrchk(cudaFree(d_v_pong));
}

void RealtimePatternSimulation::initialize_perturbation()
{
    // Create CPU buffers for initial U and V fields
    std::vector<double> u_init(Nx * Ny, 1.0);
    std::vector<double> v_init(Nx * Ny, 0.0);

    // Set a small 5x5 square in the middle to v = 1.0
    for (int y = Ny/2 - 2; y <= Ny/2 + 2; y++) {
        for (int x = Nx/2 - 2; x <= Nx/2 + 2; x++) {
            if (x >= 0 && x < Nx && y >= 0 && y < Ny) {
                v_init[y * Nx + x] = 1.0;
            }
        }
    }

    // Copy initial conditions to GPU ping buffer (active_buffer starts at 0)
    gpuErrchk(cudaMemcpy(d_u_ping, u_init.data(), Nx * Ny * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v_ping, v_init.data(), Nx * Ny * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize state
    active_buffer = 0;
    current_timestep = 0;
}

double RealtimePatternSimulation::calculate_stable_dt(double dx, double du, double dv)
{
    double dt_limit = dx*dx / (4 * std::max(du, dv));

    return 0.5 * dt_limit;
}

void RealtimePatternSimulation::step(int num_steps)
{
    for (int i = 0; i < num_steps; i++)
    {
        if (active_buffer)
        {
            compute_time_step<<<blocks, threads>>>(
                d_u_pong, d_v_pong, d_u_ping, d_v_ping, Nx, Ny,
                params.f, params.k, params.du, params.dv, dx, dt 
            );
            active_buffer = 0;
        }
        else
        {
            compute_time_step<<<blocks, threads>>>(
                d_u_ping, d_v_ping, d_u_pong, d_v_pong, Nx, Ny,
                params.f, params.k, params.du, params.dv, dx, dt 
            );
            active_buffer = 1;
        }
        current_timestep++;
    }
    gpuErrchk(cudaDeviceSynchronize());
}

void RealtimePatternSimulation::get_state(std::vector<double>& u_out, std::vector<double>& v_out)
{
    u_out.resize(Nx * Ny);
    v_out.resize(Nx * Ny);

    if (active_buffer)
    {
        gpuErrchk(cudaMemcpy(u_out.data(), d_u_pong, sizeof(double) * Nx * Ny, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(v_out.data(), d_v_pong, sizeof(double) * Nx * Ny, cudaMemcpyDeviceToHost));
    }
    else
    { 
        gpuErrchk(cudaMemcpy(u_out.data(), d_u_ping, sizeof(double) * Nx * Ny, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(v_out.data(), d_v_ping, sizeof(double) * Nx * Ny, cudaMemcpyDeviceToHost));
    }
}

void RealtimePatternSimulation::reset()
{
    initialize_perturbation();
    current_timestep = 0;
    active_buffer = 0;
}
