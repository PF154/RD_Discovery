#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include <SFML/Graphics.hpp>
#include <imgui.h>
#include <imgui-SFML.h>

#include "particle.h"
#include "rendering.h"
#include "pattern_detection.cuh"
#include "async_pattern_detector.h"

int main()
{
    // Create a 1000x1000 window
    sf::RenderWindow window(sf::VideoMode(1300, 1000), "SFML Test Window");

    if (!ImGui::SFML::Init(window))
    {
        std::cerr << "Failed to initialize ImGui" << std::endl;
        std::exit(1);
    }
    sf::Clock clock;

    float R = 100.0f;
    float G = 200.0f;
    float B = 100.0f;

    bool display_particles = true;

    // Create and set up particles.
    constexpr int num_particles = 400;
    std::vector<Particle> particles;
    particles.reserve(num_particles);

    // set up random engine
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> pos_dist(0.0, 1.0);
    std::uniform_real_distribution<double> dir_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> speed_dist(0.05, 0.15);


    for (int i = 0; i < num_particles; i++)
    {
        double f = pos_dist(gen);
        double k = pos_dist(gen);
        Vec4D position = Vec4D(f, k, 0.16, 0.08);
        double speed = speed_dist(gen);
        Vec4D direction = Vec4D(dir_dist(gen), dir_dist(gen), 0.0, 0.0);
        particles.emplace_back(Particle(position, speed, direction));
    }

    // Create thread management object
    AsyncPatternDetector detector;
    int next_request_id = 0;

    // Particle visualisation geometry
    sf::CircleShape circle(2.5);
    sf::RectangleShape hit_rect(sf::Vector2f(10, 10));
    hit_rect.setFillColor(sf::Color::Green);

    // We want a texture to display a thumbnail of discovered patterns
    // For now it will just display a color based on cursor position
    sf::RenderTexture thumbnailTexture;
    thumbnailTexture.create(150, 150);

    sf::RectangleShape thumbnail(sf::Vector2f(150, 150));

    // We want a visualization of where the user's cursor is
    sf::RectangleShape hover_rect(sf::Vector2f(10, 10));
    hover_rect.setFillColor(sf::Color::Transparent);
    hover_rect.setOutlineColor(sf::Color(3, 252, 236));
    hover_rect.setOutlineThickness(-1.0f);

    // Keep track of valid patterns
    // Eventually, this should be a more complex data structure
    std::vector<PatternResult> turing;

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

        // See if there are any discovered patterns that we can display!
        std::vector<AsyncPatternDetector::Result> results;
        if (detector.try_get_results(results, 100) > 0)
        {
            for (const auto& result : results)
            {
                for (const auto& pattern : result.patterns)
                {
                    // Come back and correct this condition when parameters are better
                    // tuned
                    if (pattern.classification == PatternType::OSCILLATING_PATTERN)
                    {
                        turing.emplace_back(pattern);
                    }
                }
            }
        }

        // Submit work every second (should really be faster than that)
        static sf::Clock detection_timer;
        if (detection_timer.getElapsedTime().asSeconds() > 1.0) {
            scan_particle_positions(particles, detector, next_request_id);
            detection_timer.restart();
        }

        sf::Time delta = clock.restart();
        ImGui::SFML::Update(window, delta);

        // Lock ImGui to bottom of screen
        ImGui::SetNextWindowPos(ImVec2(1000, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(300, 1000), ImGuiCond_Always);

        ImGuiWindowFlags flags  = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize;
        ImGui::Begin("Reaction Diffusion", nullptr, flags);

        // ImGui::Rows(2, "rows");

        ImGui::SliderFloat("R", &R, 0.0f, 255.0f);
        ImGui::SliderFloat("G", &G, 0.0f, 255.0f);
        ImGui::SliderFloat("B", &B, 0.0f, 255.0f);

        ImGui::Checkbox("Display Particles", &display_particles);

        // ImGui::NextColumn();

        // Set up and display example color
        bool valid_mouse = true;
        sf::Color thumbnailColor;
        sf::Vector2i mousePos = sf::Mouse::getPosition(window);
        if (mousePos.y > 1000 || mousePos.y < 0 || mousePos.x > 1000 || mousePos.x < 0) 
        {
            thumbnailColor = sf::Color(0, 0, 0);
            valid_mouse = false;
        }
        else
        {
            float norm_x = mousePos.x / 1000.0f;
            float norm_y = mousePos.y / 1000.0f;
            thumbnailColor = sf::Color(
                static_cast<sf::Uint32>(norm_x * 255),
                static_cast<sf::Uint32>(norm_y * 255), 
                static_cast<sf::Uint32>(255)
            );
        }

        thumbnail.setFillColor(thumbnailColor);
        thumbnailTexture.draw(thumbnail);
        ImGui::Image(thumbnailTexture, sf::Vector2f(150, 150));

        ImGui::End();

        update_particle_positions(particles, turing, delta);

        // Clear window
        window.clear(sf::Color::Black);

        // Draw turing pattern hits
        for (const PatternResult& pattern : turing)
        {
            float scale_f = pattern.params.f * 1000.0f;
            float scale_k = pattern.params.k * 1000.0f;

            double justify_f = scale_f - std::fmod(scale_f, 10.0);
            double justify_k = scale_k - std::fmod(scale_k, 10.0);

            hit_rect.setPosition(justify_f, justify_k);
            window.draw(hit_rect);
        }

        if (display_particles)
        {
            // Draw particles to screen
            sf::Color particle_color = sf::Color(static_cast<uint8_t>(R), static_cast<uint8_t>(G), static_cast<uint8_t>(B));
            circle.setFillColor(particle_color);
            for (const Particle& particle : particles)
            {
                circle.setPosition(particle.pos.f * 1000.0 - 2.5, particle.pos.k * 1000.0 - 2.5);

                // Normalize directions before drawing noses for consistent length
                double dir_magnitude = std::sqrt(
                    particle.dir.f * particle.dir.f + particle.dir.k * particle.dir.k
                );
                double norm_f = particle.dir.f / dir_magnitude;
                double norm_k = particle.dir.k / dir_magnitude;

                auto nose = create_line(
                    sf::Vector2f(particle.pos.f * 1000.0, particle.pos.k * 1000.0),
                    sf::Vector2f(
                        particle.pos.f * 1000.0 + norm_f * 10,
                        particle.pos.k * 1000.0 + norm_k * 10
                    ),
                    particle_color
                );
                window.draw(circle);
                window.draw(nose);
            }
        }

        if (valid_mouse)
        {
            float hover_x = mousePos.x - std::fmod(mousePos.x, 10.0);
            float hover_y = mousePos.y - std::fmod(mousePos.y, 10.0);

            hover_rect.setPosition(hover_x, hover_y);
            window.draw(hover_rect);
        }

        ImGui::SFML::Render(window);

        // Display
        window.display();
    }

    ImGui::SFML::Shutdown();

    return 0;
}
