#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <memory>
#include <optional>

#include <SFML/Graphics.hpp>
#include <imgui.h>
#include <imgui-SFML.h>

#include "particle.h"
#include "rendering.h"
#include "pattern_detection.cuh"
#include "async_pattern_detector.h"
#include "sim_types.h"
#include "realtime_simulation.cuh"
#include "pattern_viewer_state.h"
#include "particle_swarm_mode.h"
#include "realtime_mode.h"
#include "utilities.h"

int main()
{
    AppMode mode = AppMode::PARTICLE_SWARM;

    // Create a 1000x1000 window
    sf::RenderWindow window(sf::VideoMode(1300, 1000), "Turing Pattern Discovery");

    if (!ImGui::SFML::Init(window))
    {
        std::cerr << "Failed to initialize ImGui" << std::endl;
        std::exit(1);
    }
    sf::Clock clock;

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

    // Keep track of valid patterns
    // Eventually, this should be a more complex data structure
    std::vector<PatternResult> turing;

    std::optional<PatternViewerState> realtime_pattern;

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
            if (event.type == sf::Event::MouseButtonPressed &&
                event.mouseButton.button == sf::Mouse::Left) {
                int clicked_idx = find_pattern_under_mouse(sf::Mouse::getPosition(window), turing);
                if (clicked_idx >= 0 && clicked_idx < turing.size()) {
                    realtime_pattern.emplace(PatternViewerState{
                        std::make_unique<RealtimePatternSimulation>(turing[clicked_idx].params),
                        sf::Texture(),
                        30
                    });
                    mode = AppMode::REAL_TIME_PATTERN;
                }
            }
        }

        if (mode == AppMode::PARTICLE_SWARM)
        {
            run_particle_swarm(window, turing, detector, particles, clock);
        }
        else
        {
            run_real_time_sim(window, mode, clock, realtime_pattern);
        }
    }

    ImGui::SFML::Shutdown();

    return 0;
}

