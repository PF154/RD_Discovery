#include <iostream>

#include <memory>
#include <cstdint>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <cufft.h>
#include <algorithm>

#include <SFML/Graphics.hpp>
#include <imgui.h>
#include <imgui-SFML.h>

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

struct Vec4D
{
    Vec4D(double f, double k, double du, double dv) 
        : f(f), k(k), du(du), dv(dv) {};
    double f = 0.0;
    double k = 0.0;
    double du = 0.0;
    double dv = 0.0;
};

struct Particle
{
    Particle(Vec4D pos, double speed, Vec4D dir)
        : pos(pos), speed(speed), dir(dir) {};
    Vec4D pos;
    double speed;
    Vec4D dir;
};

bool is_periodic(Vec4D& position)
{
    // Eventually, this will do some better computation to determine
    // if the parameters actually result in a turing pattern.
    // For now, it will just return true 0.0001 percent of the time.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> pos_dist(0.0, 1.0);

    if (pos_dist(gen) < 0.0001) return true;
    return false;
}

void scan_particle_positions(std::vector<Particle>& particles, std::vector<Vec4D>& turing)
{  
    for (Particle& particle: particles)
    {
        if (is_periodic(particle.pos)) turing.push_back(particle.pos);
    }
}

void update_particle_positions(std::vector<Particle>& particles, const sf::Time& delta)
{
    for (Particle& particle: particles)
    {
        particle.pos.f += particle.dir.f * particle.speed * delta.asSeconds();
        particle.pos.k += particle.dir.k * particle.speed * delta.asSeconds();

        // Perioidic
        particle.pos.f = std::fmod(particle.pos.f + 1.0, 1.0);
        particle.pos.k = std::fmod(particle.pos.k + 1.0, 1.0);
    }
}

// Helper function to draw particle noses
sf::VertexArray create_line(sf::Vector2f start, sf::Vector2f end, sf::Color color)
{
    sf::VertexArray line(sf::Lines, 2);
    line[0] = sf::Vertex(start, color);
    line[1] = sf::Vertex(end, color);
    return line;
}


int main()
{
    // Create a 500x500 window
    sf::RenderWindow window(sf::VideoMode(500, 700), "SFML Test Window");

    if (!ImGui::SFML::Init(window))
    {
        std::cerr << "Failed to initialize ImGui" << std::endl;
        std::exit(1);
    }
    sf::Clock clock;

    float R = 100.0f;
    float G = 200.0f;
    float B = 100.0f;

    // Create and set up particles.
    constexpr int num_particles = 400;
    std::vector<Particle> particles;
    particles.reserve(num_particles);

    // set up random engine
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> pos_dist(0.0, 1.0);
    std::uniform_real_distribution<double> dir_dist(-1.0, 1.0);    
    std::uniform_real_distribution<double> speed_dist(0.25, 0.5);    
    

    for (int i = 0; i < num_particles; i++)
    {
        double f = pos_dist(gen);
        double k = pos_dist(gen);
        Vec4D position = Vec4D(f, k, 0.0, 0.0);
        double speed = speed_dist(gen);
        Vec4D direction = Vec4D(dir_dist(gen), dir_dist(gen), 0.0, 0.0);
        particles.emplace_back(Particle(position, speed, direction));
    }

    // Particle visualisation geometry
    sf::CircleShape circle(2.5);
    sf::RectangleShape hit_rect(sf::Vector2f(5, 5));
    hit_rect.setFillColor(sf::Color::Green);

    // Keep track of valid patterns
    // Eventually, this should be a more complex data structure
    std::vector<Vec4D> turing;

    // Main loop
    while (window.isOpen())
    {
        // Handle events
        sf::Event event;
        while (window.pollEvent(event))
        {
            ImGui::SFML::ProcessEvent(window, event);
            if (event.type == sf::Event::Closed)
                window.close();
        }

        sf::Time delta = clock.restart();
        ImGui::SFML::Update(window, delta);

        // Lock ImGui to bottom of screen
        ImGui::SetNextWindowPos(ImVec2(0, 500), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(500, 200), ImGuiCond_Always);

        ImGuiWindowFlags flags  = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize;
        ImGui::Begin("Reaction Diffusion", nullptr, flags);

        ImGui::SliderFloat("R", &R, 0.0f, 255.0f);
        ImGui::SliderFloat("G", &G, 0.0f, 255.0f);
        ImGui::SliderFloat("B", &B, 0.0f, 255.0f);

        ImGui::End();

        update_particle_positions(particles, delta);
        scan_particle_positions(particles, turing);

        // Clear window
        window.clear(sf::Color::Black);

        // Draw turing pattern hits
        for (const Vec4D& pos : turing)
        {
            hit_rect.setPosition(pos.f * 500.0, pos.k * 500.0);
            window.draw(hit_rect);
        }

        // Draw particles to screen
        sf::Color particle_color = sf::Color(static_cast<uint8_t>(R), static_cast<uint8_t>(G), static_cast<uint8_t>(B));
        circle.setFillColor(particle_color);
        for (const Particle& particle : particles)
        {
            circle.setPosition(particle.pos.f * 500.0 - 2.5, particle.pos.k * 500.0 - 2.5);

            // Normalize directions before drawing noses for consisten length
            double dir_magnitude = std::sqrt(
                particle.dir.f * particle.dir.f + particle.dir.k * particle.dir.k
            );
            double norm_f = particle.dir.f / dir_magnitude;
            double norm_k = particle.dir.k / dir_magnitude;

            auto nose = create_line(
                sf::Vector2f(particle.pos.f * 500.0, particle.pos.k * 500.0),
                sf::Vector2f(
                    particle.pos.f * 500.0 + norm_f * 10,
                    particle.pos.k * 500.0 + norm_k * 10
                ),
                particle_color
            );
            window.draw(circle);
            window.draw(nose);
        }

        ImGui::SFML::Render(window);

        // Display
        window.display();
    }

    ImGui::SFML::Shutdown();

    return 0;
}