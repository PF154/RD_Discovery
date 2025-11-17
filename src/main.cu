#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include <SFML/Graphics.hpp>
#include <imgui.h>
#include <imgui-SFML.h>

#include "particle.h"
#include "rendering.h"
#include "cuda_kernels.cuh"

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
    std::uniform_real_distribution<double> speed_dist(0.15, 0.25);


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

        update_particle_positions(particles, turing, delta);
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

            // Normalize directions before drawing noses for consistent length
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
