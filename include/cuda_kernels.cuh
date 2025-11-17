#pragma once

#include "definitions.cuh"
#include <cstdint>

#define Nx 100
#define Ny 100

// CUDA constant memory
extern __constant__ GlobalConstants globalConsts;

// CUDA kernel declarations
__global__ void compute_time_step(double* u_data_in, double* v_data_in, double* u_data_out, double* v_data_out);
__global__ void map_to_rgba(const double* u_data, const double* v_data, uint8_t* out_pixels);

// Host functions
extern "C" void setGlobalConsts(GlobalConstants *h_consts);
