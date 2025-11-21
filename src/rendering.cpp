#include "rendering.h"
#include "pattern_detection.cuh"
#include <SFML/Graphics.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdint>

sf::VertexArray create_line(sf::Vector2f start, sf::Vector2f end, sf::Color color)
{
    sf::VertexArray line(sf::Lines, 2);
    line[0] = sf::Vertex(start, color);
    line[1] = sf::Vertex(end, color);
    return line;
}

void draw_axes(sf::RenderWindow& window, float min_f, float max_f, float min_k, float max_k)
{
    const float AXIS_SIZE = 1000.0f;
    const float AXIS_OFFSET = 10.0f;
    const int NUM_TICKS = 10;
    const float TICK_LENGTH = 10.0f;

    // Draw main axes
    sf::VertexArray f_axis = create_line(
        sf::Vector2f(AXIS_OFFSET, AXIS_OFFSET),
        sf::Vector2f(AXIS_SIZE, AXIS_OFFSET),
        sf::Color::White
    );
    window.draw(f_axis);

    sf::VertexArray k_axis = create_line(
        sf::Vector2f(AXIS_OFFSET, AXIS_OFFSET),
        sf::Vector2f(AXIS_OFFSET, AXIS_SIZE),
        sf::Color::White
    );
    window.draw(k_axis);

    // Load font (static to load only once)
    static sf::Font font;
    static bool font_loaded = false;
    if (!font_loaded) {
        // Try common font paths (Linux, then Windows)
        if (font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") ||
            font.loadFromFile("/usr/share/fonts/TTF/DejaVuSans.ttf") ||
            font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf")) {
            font_loaded = true;
        }
    }

    // Draw f-axis ticks (vertical ticks along horizontal axis at top)
    for (int i = 0; i <= NUM_TICKS; i++) {
        float t = static_cast<float>(i) / NUM_TICKS;
        float x = t * AXIS_SIZE;  // Evenly space across full screen width
        float f_value = min_f + t * (max_f - min_f);

        // Draw tick mark (pointing down)
        sf::VertexArray tick = create_line(
            sf::Vector2f(x, AXIS_OFFSET),
            sf::Vector2f(x, AXIS_OFFSET + TICK_LENGTH),
            sf::Color::White
        );
        window.draw(tick);

        // Draw label
        if (font_loaded) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << f_value;

            sf::Text label;
            label.setFont(font);
            label.setString(oss.str());
            label.setCharacterSize(10);
            label.setFillColor(sf::Color::White);

            // Center the text below the tick
            sf::FloatRect bounds = label.getLocalBounds();
            label.setPosition(x - bounds.width / 2, AXIS_OFFSET + TICK_LENGTH + 2);
            window.draw(label);
        }
    }

    // Draw k-axis ticks (horizontal ticks along vertical axis on left)
    for (int i = 0; i <= NUM_TICKS; i++) {
        float t = static_cast<float>(i) / NUM_TICKS;
        float y = t * AXIS_SIZE;  // Evenly space across full screen height
        float k_value = min_k + t * (max_k - min_k);

        // Draw tick mark (pointing right)
        sf::VertexArray tick = create_line(
            sf::Vector2f(AXIS_OFFSET, y),
            sf::Vector2f(AXIS_OFFSET + TICK_LENGTH, y),
            sf::Color::White
        );
        window.draw(tick);

        // Draw label
        if (font_loaded) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << k_value;

            sf::Text label;
            label.setFont(font);
            label.setString(oss.str());
            label.setCharacterSize(10);
            label.setFillColor(sf::Color::White);

            // Position to the right of the axis
            sf::FloatRect bounds = label.getLocalBounds();
            label.setPosition(AXIS_OFFSET - bounds.width + 45, y - bounds.height / 2 - 2);
            window.draw(label);
        }
    }
}

sf::Image image_from_pattern_data(const PatternResult& pattern) 
{
    const int Nx = 100;
    const int Ny = 100;

    sf::Image image;
    image.create(Nx, Ny);

    for (int i = 0; i < Nx * Ny; i++)
    {
        double u = pattern.u_final[i];
        double v = pattern.v_final[i];

        uint8_t r = 0;
        uint8_t g = static_cast<uint8_t>(std::clamp(v * 255.0, 0.0, 255.0));
        uint8_t b = static_cast<uint8_t>(std::clamp(u * 255.0, 0.0, 255.0));

        int x = i % Nx;
        int y = i / Nx;
        image.setPixel(x, y, sf::Color(r, g, b));
    }

    return image;
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
        out.put(data[i * 4 + 0]);
        out.put(data[i * 4 + 1]);
        out.put(data[i * 4 + 2]);
    }

    out.close();
}
