#pragma once

#include <vector>
#include <optional>
#include <SFML/Graphics.hpp>

#include "sim_types.h"
#include "pattern_viewer_state.h"

/**
 * Generate an sf::Image from pattern data
 * 
 * @param u Vector of doubles of size Nx*Ny, representing the concentration of chemical u
 * @param v Vector of doubles of size Nx*Ny, representing the concentration of chemical v
 * @param Nx Width of the image
 * @param Ny Height of the image
 * @return sf::Image generated from the input pattern data
 */
sf::Image create_image_from_simulation(
    const std::vector<double>& u,
    const std::vector<double>& v,
    int Nx, int Ny
);

/**
 * The main logic for real time sumulation mode, where users can watch a single pattern develop in real time.
 * 
 * @param window The sf::RenderWindow that the simulation should be rendered to
 * @param mode The AppMode variable in the calling scope (modified by "Return to Particle Swarm" button)
 * @param clock sf:Clock to track time elapsed
 * @param pattern_viewer PatternViewerState to be used
 */
void run_real_time_sim(
    sf::RenderWindow& window,
    AppMode& mode,
    sf::Clock& clock,
    std::optional<PatternViewerState>& pattern_viewer // Note: Why is this optional?? Seems like old code that should be changed
);