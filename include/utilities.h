#pragma once

#include <SFML/Graphics.hpp>
#include <vector>

#include "sim_types.h"
#include "pattern_detection.cuh"

// Transform parameter space coordinates to screen space coordinates
sf::Vector2f param_to_screen(double param_f, double param_k, const FKExtents& extents);

// If extent selection is inverted, flip it back
void correct_extents(FKExtents& extents);

int find_pattern_under_mouse(sf::Vector2i mousePos, std::vector<PatternResult>& turing, const FKExtents& extents);