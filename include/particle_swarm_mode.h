#pragma once

#include <SFML/Graphics.hpp>

#include "sim_types.h"
#include "async_pattern_detector.h"
#include "particle.h"

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

void update_pattern_display(
    int hovered_pattern_idx, 
    const std::vector<PatternResult>& turing, 
    sf::Texture& pattern_texture
);