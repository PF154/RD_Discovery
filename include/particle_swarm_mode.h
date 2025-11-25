#pragma once

#include <SFML/Graphics.hpp>

#include "sim_types.h"
#include "async_pattern_detector.h"
#include "particle.h"

/**
 * Main particle swarm simulation logic
 *
 * @param window Window to render particle swarm to
 * @param turing Vector of PatternResult objects representing discovered patterns
 * @param detector AsyncPatternDetector to handle sending work to GPU
 * @param particles Vector of Particle objects whose positions attempt to generate patterns
 * @param clock sf::Clock to update frames
 * @param extents FKExtents object which represents the extents of the current window
 * @param selection_state SelectionState object used to track whether the user is drag selecting
 * @param reset_extents Send signal from main to simulation loop if extents should be reset
*/
void run_particle_swarm(
    sf::RenderWindow& window,
    std::vector<PatternResult>& turing,
    AsyncPatternDetector& detector,
    std::vector<Particle>& particles,
    sf::Clock& clock,
    FKExtents& extents,
    const SelectionState& selection_state,
    bool& reset_extents
);

/**
 * Handles the pattern thumbnail in the lower right of the screen
 * 
 * @param hovered_pattern_idx The index of the pattern in turing which is being hovered over
 * @param turing Vector of PatternResults to be pulled from
 * @param pattern_texture sf::Texture which will be written to once the thumbnail is generated
 */
void update_pattern_display(
    int hovered_pattern_idx, 
    const std::vector<PatternResult>& turing, 
    sf::Texture& pattern_texture
);