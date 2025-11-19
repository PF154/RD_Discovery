#include "utilities.h"
#include "pattern_detection.cuh"
#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>

int find_pattern_under_mouse(sf::Vector2i mousePos, std::vector<PatternResult>& turing)
{
    // Justify mouse position to 10-pixel grid (same as hit_rect rendering)
    float mouse_grid_x = mousePos.x - std::fmod(mousePos.x, 10.0);
    float mouse_grid_y = mousePos.y - std::fmod(mousePos.y, 10.0);

    for (int i = 0; i < turing.size(); i++)
    {
        // Scale pattern position to pixels
        float scale_f = turing[i].params.f * 1000.0f;
        float scale_k = turing[i].params.k * 1000.0f;

        // Justify pattern position to 10-pixel grid
        float pattern_grid_x = scale_f - std::fmod(scale_f, 10.0);
        float pattern_grid_y = scale_k - std::fmod(scale_k, 10.0);

        // Check if in same 10-pixel box
        if (mouse_grid_x == pattern_grid_x && mouse_grid_y == pattern_grid_y) {
            return i;
        }
    }
    return -1;  // No pattern under mouse
}