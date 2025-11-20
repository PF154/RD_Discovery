#include "utilities.h"
#include "pattern_detection.cuh"
#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>

sf::Vector2f param_to_screen(double param_f, double param_k, const FKExtents& extents)
{
    // Normalize to [0, 1] based on extents
    float norm_f = (param_f - extents.min_f) / (extents.max_f - extents.min_f);
    float norm_k = (param_k - extents.min_k) / (extents.max_k - extents.min_k);

    // Scale to screen coordinates [0, 1000]
    return sf::Vector2f(norm_f * 1000.0f, norm_k * 1000.0f);
}

int find_pattern_under_mouse(sf::Vector2i mousePos, std::vector<PatternResult>& turing, const FKExtents& extents)
{
    // Justify mouse position to 10-pixel grid (same as hit_rect rendering)
    float mouse_grid_x = mousePos.x - std::fmod(mousePos.x, 10.0);
    float mouse_grid_y = mousePos.y - std::fmod(mousePos.y, 10.0);

    for (int i = 0; i < turing.size(); i++)
    {
        // Transform pattern position from parameter space to screen space using extents
        sf::Vector2f screen_pos = param_to_screen(turing[i].params.f, turing[i].params.k, extents);

        // Justify pattern position to 10-pixel grid
        float pattern_grid_x = screen_pos.x - std::fmod(screen_pos.x, 10.0);
        float pattern_grid_y = screen_pos.y - std::fmod(screen_pos.y, 10.0);

        // Check if in same 10-pixel box
        if (mouse_grid_x == pattern_grid_x && mouse_grid_y == pattern_grid_y) {
            return i;
        }
    }
    return -1;  // No pattern under mouse
}

void correct_extents(FKExtents& extents)
{
    if (extents.max_f < extents.min_f) std::swap(extents.max_f, extents.min_f);
    if (extents.max_k < extents.min_k) std::swap(extents.max_k, extents.min_k);
}
