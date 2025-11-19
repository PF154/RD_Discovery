#pragma once

#include <SFML/Graphics.hpp>
#include <vector>

#include "sim_types.h"
#include "pattern_detection.cuh"

int find_pattern_under_mouse(sf::Vector2i mousePos, std::vector<PatternResult>& turing);