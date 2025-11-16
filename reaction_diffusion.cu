#include <iostream>

#include <memory>
#include <cstdint>
#include <random>
#include <fstream>
#include <string>
#include <cufft.h>
#include <algorithm>

#include <SFML/Graphics.hpp>

#include "definitions.cuh"

#define Nx 100
#define Ny 100


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

int main()
{
    // Create a 500x500 window
    sf::RenderWindow window(sf::VideoMode(500, 500), "SFML Test Window");

    // Create a green rectangle in the middle
    sf::RectangleShape greenBox(sf::Vector2f(200.f, 200.f));
    greenBox.setFillColor(sf::Color::Green);
    greenBox.setPosition(150.f, 150.f);  // Center it (500-200)/2 = 150

    // Main loop
    while (window.isOpen())
    {
        // Handle events
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Clear window
        window.clear(sf::Color::Black);

        // Draw the green box
        window.draw(greenBox);

        // Display
        window.display();
    }

    return 0;
}