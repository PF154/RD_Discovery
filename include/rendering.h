#pragma once

#include <SFML/Graphics.hpp>
#include <string>
#include <cstdint>

// Helper function to create a line between two points
sf::VertexArray create_line(sf::Vector2f start, sf::Vector2f end, sf::Color color);

// Write image data to PPM file
void write_ppm(const std::string& filename, const uint8_t* data, int width, int height);
