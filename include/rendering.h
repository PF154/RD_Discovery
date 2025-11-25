#pragma once

#include "pattern_detection.cuh"

#include <SFML/Graphics.hpp>
#include <string>
#include <cstdint>

/**
 * Helper function to create a line between two points
 * 
 * @param start Start of the line in screen space
 * @param end End of the line in screen space
 * @param color Intended color of line
 * @return sf::VertexArray of the specified line
*/
sf::VertexArray create_line(sf::Vector2f start, sf::Vector2f end, sf::Color color);

/**
 * Draw coordinate axes the the screen
 * 
 * @param window sf::RenderWindow for the axes to be drawn to
 * @param min_f minimum f axis value
 * @param max_f maximum f axis value
 * @param min_k minimum k axis value
 * @param max_k maximum k axis value
 */
void draw_axes(sf::RenderWindow& window, float min_f, float max_f, float min_k, float max_k);

// Can this be removed?
sf::Image image_from_pattern_data(const PatternResult& pattern);

/**
 * For debugging, write pattern data to a PPM for manual checking
 * 
 * @param filename Name out output file
 * @param data width*height long array of data representing concentrations of one chemical
 * @param width Width of data
 * @param height Height of data
 */
void write_ppm(const std::string& filename, const uint8_t* data, int width, int height);
