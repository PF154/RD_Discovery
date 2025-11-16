#include <iostream>
#include "SFML-3.0.0/include/SFML/Graphics.hpp"

#include <memory>
#include <cstdint>
#include <random>
#include <fstream>
#include <string> 

#include "definitions.cuh"

#define Nx 500
#define Ny 500


__constant__  GlobalConstants globalConsts;

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


void write_ppm(const std::string& filename, const uint8_t* data, int width, int height) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // PPM header
    out << "P6\n" << width << " " << height << "\n255\n";

    // Write RGB values (ignore alpha if present)
    for (int i = 0; i < (width * height); ++i) {
        out.put(data[i * 4 + 0]); // Red
        out.put(data[i * 4 + 1]); // Green
        out.put(data[i * 4 + 2]); // Blue
    }

    out.close();
}

extern "C" void generate_image_based_on_params(double Du, double Dv, double F, double k, int timesteps)
{
    GlobalConstants h_consts;
    h_consts.Du = Du;
    h_consts.Dv = Dv;
    h_consts.F = F;
    h_consts.k = k;

    setGlobalConsts(&h_consts);

    const unsigned int total_array_size(Nx * Ny);

    std::unique_ptr<double[]> h_u_data = std::make_unique<double[]>(total_array_size);
    std::unique_ptr<double[]> h_v_data = std::make_unique<double[]>(total_array_size);

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 1);

    for (int i=0; i<Nx*Ny; i++)
    {
        h_u_data[i] = 1.0;
        h_v_data[i] = 0.0;
    }

    // Square
    for (int y = Ny/2 - 10; y < Ny/2 + 10; y++)
    {
        for (int x = Nx/2 - 10; x < Nx/2 + 10; x++)
        {
            h_u_data[y * Nx + x] = 0.5;
            h_v_data[y * Nx + x] = 0.25;
        }
    }
    
    size_t pixelBytes = total_array_size * 4 * sizeof(uint8_t);

    std::unique_ptr<uint8_t[]> h_pixels = std::make_unique<uint8_t[]>(pixelBytes);

    uint8_t *d_pixels;
    gpuErrchk(cudaMalloc(&d_pixels, pixelBytes));

    double *d_u_ping, *d_v_ping, *d_u_pong, *d_v_pong;
    size_t bytes = sizeof(double) * total_array_size;

    gpuErrchk(cudaMalloc(&d_u_ping, bytes));
    gpuErrchk(cudaMalloc(&d_v_ping, bytes));
    gpuErrchk(cudaMalloc(&d_u_pong, bytes));
    gpuErrchk(cudaMalloc(&d_v_pong, bytes));

    gpuErrchk(cudaMemcpy(d_u_ping, h_u_data.get(), bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v_ping, h_v_data.get(), bytes, cudaMemcpyHostToDevice));

    for (int i=0; i<timesteps; i++)
    {
        compute_time_step<<<int(Nx*Ny/32) + 1, 32>>>(d_u_ping, d_v_ping, d_u_pong, d_v_pong);
        gpuErrchk(cudaDeviceSynchronize());

        std::swap(d_u_ping, d_u_pong);
        std::swap(d_v_ping, d_v_pong);
    }

    map_to_rgba<<<int(Nx*Ny/32) + 1, 32>>>(d_u_pong, d_v_pong, d_pixels);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_pixels.get(), d_pixels, pixelBytes, cudaMemcpyDeviceToHost));

    std::string filename = std::to_string(Du) + "_" + std::to_string(Dv) + "_" + std::to_string(F) + "_" + std::to_string(k) + ".ppm";
    
    write_ppm(filename, h_pixels.get(), Nx, Ny);

    cudaFree(d_pixels);

    cudaFree(d_u_ping);
    cudaFree(d_v_ping);
    cudaFree(d_u_pong);
    cudaFree(d_v_pong);
}

// int main()
// {
    
// }

int main()
{
    const unsigned int total_array_size(Nx * Ny);

    std::unique_ptr<double[]> h_u_data = std::make_unique<double[]>(total_array_size);
    std::unique_ptr<double[]> h_v_data = std::make_unique<double[]>(total_array_size);

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 1);

    for (int i=0; i<Nx*Ny; i++)
    {
        h_u_data[i] = 1.0;
        h_v_data[i] = 0.0;
    }

    // Square
    for (int y = Ny/2 - 10; y < Ny/2 + 10; y++)
    {
        for (int x = Nx/2 - 10; x < Nx/2 + 10; x++)
        {
            // h_u_data[y * Nx + x] = dist(e2);
            // h_v_data[y * Nx + x] = dist(e2);
            h_u_data[y * Nx + x] = 0.5;
            h_v_data[y * Nx + x] = 0.25;
        }
    }

    sf::RenderWindow window(sf::VideoMode(sf::Vector2u{Nx, Ny}), "Reaction Diffusion Simulation");
    window.setFramerateLimit(60);
    
    size_t pixelBytes = total_array_size * 4 * sizeof(uint8_t);

    std::unique_ptr<uint8_t[]> h_pixels = std::make_unique<uint8_t[]>(pixelBytes);

    uint8_t *d_pixels;
    gpuErrchk(cudaMalloc(&d_pixels, pixelBytes));

    double *d_u_ping, *d_v_ping, *d_u_pong, *d_v_pong;
    size_t bytes = sizeof(double) * total_array_size;

    gpuErrchk(cudaMalloc(&d_u_ping, bytes));
    gpuErrchk(cudaMalloc(&d_v_ping, bytes));
    gpuErrchk(cudaMalloc(&d_u_pong, bytes));
    gpuErrchk(cudaMalloc(&d_v_pong, bytes));

    gpuErrchk(cudaMemcpy(d_u_ping, h_u_data.get(), bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v_ping, h_v_data.get(), bytes, cudaMemcpyHostToDevice));

    sf::Texture texture(sf::Vector2u{Nx, Ny});
    sf::Sprite sprite(texture);

    const int timesteps_per_frame = 10;
    while(window.isOpen())
    {
        while(auto maybeEvent = window.pollEvent())
        {
            if(maybeEvent->is<sf::Event::Closed>())
            {
                window.close();
            }
        }

        for (int i=0; i<timesteps_per_frame; i++)
        {
            compute_time_step<<<int(Nx*Ny/32) + 1, 32>>>(d_u_ping, d_v_ping, d_u_pong, d_v_pong);
            gpuErrchk(cudaDeviceSynchronize());

            std::swap(d_u_ping, d_u_pong);
            std::swap(d_v_ping, d_v_pong);
        }

        map_to_rgba<<<int(Nx*Ny/32) + 1, 32>>>(d_u_pong, d_v_pong, d_pixels);
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaMemcpy(h_pixels.get(), d_pixels, pixelBytes, cudaMemcpyDeviceToHost));
        texture.update(h_pixels.get());

        window.clear();
        window.draw(sprite);
        window.display();
    }

    cudaFree(d_pixels);

    cudaFree(d_u_ping);
    cudaFree(d_v_ping);
    cudaFree(d_u_pong);
    cudaFree(d_v_pong);

    return 0; 
}