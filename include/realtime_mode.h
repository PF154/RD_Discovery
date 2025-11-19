#pragma once

#include <vector>
#include <optional>
#include <SFML/Graphics.hpp>

#include "sim_types.h"
#include "pattern_viewer_state.h"

sf::Image create_image_from_simulation(
    const std::vector<double>& u,
    const std::vector<double>& v,
    int Nx, int Ny
);

void run_real_time_sim(
    sf::RenderWindow& window,
    AppMode& mode,
    sf::Clock& clock,
    std::optional<PatternViewerState>& pattern_viewer
);