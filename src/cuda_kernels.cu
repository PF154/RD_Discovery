#include "cuda_kernels.cuh"
#include <cmath>

__constant__ GlobalConstants globalConsts;

extern "C" void setGlobalConsts(GlobalConstants *h_consts)
{
    gpuErrchk(cudaMemcpyToSymbol(globalConsts, h_consts, sizeof(GlobalConstants)));
}

static __inline__ __device__ int idx(int x, int y) {
    return y * Nx + x;
}

static __inline__ __device__ double laplace(const double* data, int x, int y) {
    int left = (x == 0) ? Nx - 1 : x - 1;
    int right = (x == Nx - 1) ? 0 : x + 1;
    int up = (y == 0) ? Ny - 1 : y - 1;
    int down = (y == Ny - 1) ? 0 : y + 1;

    return data[idx(left, y)] + data[idx(right, y)] + data[idx(x, up)] + data[idx(x, down)] - 4.0 * data[idx(x, y)];
}

__global__ void compute_time_step(double* u_data_in, double* v_data_in, double* u_data_out, double* v_data_out)
{
    const int tID = threadIdx.x + blockDim.x * blockIdx.x;
    if (tID >= Nx * Ny)
    {
        return;
    }
    const unsigned int ix(tID % Nx), iy(int(tID / Nx));

    const double u = u_data_in[tID];
    const double v = v_data_in[tID];


    u_data_out[tID] = u + globalConsts.dt * (globalConsts.Du * laplace(u_data_in, ix, iy) - u*v*v + globalConsts.F * (1 - u));
    v_data_out[tID] = v + globalConsts.dt * (globalConsts.Dv * laplace(v_data_in, ix, iy) + u*v*v - (globalConsts.F + globalConsts.k) * v);
}

__global__ void map_to_rgba(const double* u_data, const double* v_data, uint8_t* out_pixels)
{
    int tID = threadIdx.x + blockDim.x * blockIdx.x;
    if (tID >= Nx * Ny)
    {
        return;
    }

    double u = u_data[tID];
    double v = v_data[tID];

    int base = 4 * tID;
    out_pixels[base + 0] = 0;
    out_pixels[base + 1] = int(u * 255);
    out_pixels[base + 2] = int(v * 255);
    out_pixels[base + 3] = 255;
}
