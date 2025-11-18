#include "rendering.h"
#include <fstream>
#include <iostream>

sf::VertexArray create_line(sf::Vector2f start, sf::Vector2f end, sf::Color color)
{
    sf::VertexArray line(sf::Lines, 2);
    line[0] = sf::Vertex(start, color);
    line[1] = sf::Vertex(end, color);
    return line;
}

void draw_axes(sf::RenderWindow& window)
{
    sf::VertexArray f_axis = create_line(sf::Vector2f(3, 3), sf::Vector2f(1000, 3), sf::Color::White);
    window.draw(f_axis);
    sf::VertexArray k_axis = create_line(sf::Vector2f(3, 3), sf::Vector2f(3, 1000), sf::Color::White);
    window.draw(k_axis);
}

void write_ppm(const std::string& filename, const uint8_t* data, int width, int height) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // PPM header
    out << "P6\n" << width << " " << height << "\n255\n";

    // Write RGB values (ignore alpha if present)
    for (int i = 0; i < (width * height); ++i) {
        out.put(data[i * 4 + 0]); // Red
        out.put(data[i * 4 + 1]); // Green
        out.put(data[i * 4 + 2]); // Blue
    }

    out.close();
}
