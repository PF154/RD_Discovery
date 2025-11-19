#pragma once

#include <memory>
#include <SFML/Graphics.hpp>

#include "realtime_simulation.cuh"

struct PatternViewerState {
    std::unique_ptr<RealtimePatternSimulation> sim;
    sf::Texture display_texture;
    int steps_per_frame = 30;
};