#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <future>

#include <SFML/Graphics.hpp>
#include <imgui.h>
#include <imgui-SFML.h>

#include "particle.h"
#include "rendering.h"
#include "pattern_detection.cuh"
#include "async_pattern_detector.h"

// Pattern display state values
enum PatternDisplayState {
    UNINITIALIZED = -2,
    SHOWING_BLACK = -1
    // Values >= 0 represent pattern indices
};

int find_pattern_under_mouse(sf::Vector2i mousePos, std::vector<PatternResult>& turing);
void update_pattern_display(int hovered_pattern_idx, const std::vector<PatternResult>& turing, sf::Texture& pattern_texture);

template<typename T>
bool future_is_ready(const std::future<T>& f);

int main()
{
    // Create a 1000x1000 window
    sf::RenderWindow window(sf::VideoMode(1300, 1000), "Turing Pattern Discovery");

    if (!ImGui::SFML::Init(window))
    {
        std::cerr << "Failed to initialize ImGui" << std::endl;
        std::exit(1);
    }
    sf::Clock clock;

    float R = 100.0f;
    float G = 200.0f;
    float B = 100.0f;

    bool display_particles = true;
    bool display_axes = true;

    float du = 0.16;
    float dv = 0.08;

    // Create and set up particles.
    constexpr int num_particles = 400;
    std::vector<Particle> particles;
    particles.reserve(num_particles);

    // set up random engine
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> pos_dist(0.0, 1.0);
    std::uniform_real_distribution<double> dir_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> speed_dist(0.05, 0.15);


    for (int i = 0; i < num_particles; i++)
    {
        double f = pos_dist(gen);
        double k = pos_dist(gen);
        Vec4D position = Vec4D(f, k, 0.16, 0.08);
        double speed = speed_dist(gen);
        Vec4D direction = Vec4D(dir_dist(gen), dir_dist(gen), 0.0, 0.0);
        particles.emplace_back(Particle(position, speed, direction));
    }

    // Create thread management object
    AsyncPatternDetector detector;
    int next_request_id = 0;

    // Particle visualisation geometry
    sf::CircleShape circle(2.5);
    sf::RectangleShape hit_rect(sf::Vector2f(10, 10));
    hit_rect.setFillColor(sf::Color::Green);

    // We want a texture to display a thumbnail of discovered patterns
    // For now it will just display a color based on cursor position
    sf::RenderTexture thumbnailTexture;
    thumbnailTexture.create(100, 100);

    sf::RectangleShape thumbnail(sf::Vector2f(250, 250));

    // We want a visualization of where the user's cursor is
    sf::RectangleShape hover_rect(sf::Vector2f(10, 10));
    hover_rect.setFillColor(sf::Color::Transparent);
    hover_rect.setOutlineColor(sf::Color(3, 252, 236));
    hover_rect.setOutlineThickness(-1.0f);

    // Keep track of valid patterns
    // Eventually, this should be a more complex data structure
    std::vector<PatternResult> turing;

    // Pattern display state (static to persist across frames)
    static sf::Texture pattern_texture;

    // Main loop
    while (window.isOpen())
    {
        int hovered_pattern_idx = find_pattern_under_mouse(sf::Mouse::getPosition(window), turing);
        update_pattern_display(hovered_pattern_idx, turing, pattern_texture);

        // Handle events
        sf::Event event;
        while (window.pollEvent(event))
        {
            ImGui::SFML::ProcessEvent(window, event);
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // See if there are any discovered patterns that we can display!
        std::vector<AsyncPatternDetector::Result> results;
        if (detector.try_get_results(results, 100) > 0)
        {
            for (const auto& result : results)
            {
                for (const auto& pattern : result.patterns)
                {
                    // Come back and correct this condition when parameters are better
                    // tuned
                    if (pattern.classification == PatternType::OSCILLATING_PATTERN)
                    {
                        turing.emplace_back(pattern);
                    }
                }
            }
        }

        // Submit work every second (should really be faster than that)
        static sf::Clock detection_timer;
        if (detection_timer.getElapsedTime().asSeconds() > 1.0) {
            scan_particle_positions(particles, detector, next_request_id);
            detection_timer.restart();
        }

        sf::Time delta = clock.restart();
        ImGui::SFML::Update(window, delta);

        // Lock ImGui to bottom of screen
        ImGui::SetNextWindowPos(ImVec2(1000, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(300, 1000), ImGuiCond_Always);

        ImGuiWindowFlags flags  = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize;
        ImGui::Begin("Reaction Diffusion", nullptr, flags);

        // ImGui::Rows(2, "rows");

        ImGui::SliderFloat("R", &R, 0.0f, 255.0f);
        ImGui::SliderFloat("G", &G, 0.0f, 255.0f);
        ImGui::SliderFloat("B", &B, 0.0f, 255.0f);

        ImGui::SliderFloat("Du", &du, 0.0f, 1.0f);
        ImGui::SliderFloat("Dv", &dv, 0.0f, 1.0f);

        ImGui::Checkbox("Particles", &display_particles);
        ImGui::Checkbox("Axes", &display_axes);

        // ImGui::NextColumn();

        // Add any other widgets here

        // Add any other widgets above here

        // Display pattern texture (updated by update_pattern_display)
        bool valid_mouse = true;
        sf::Vector2i mousePos = sf::Mouse::getPosition(window);
        float available_width = ImGui::GetContentRegionAvail().x;
        float available_height = ImGui::GetContentRegionAvail().y;
        float offset_x = (available_width - 250) * 0.5f;
        float offset_y = (available_height - 275);

        if (offset_x > 0.0f) 
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset_x);
        if (offset_x > 0.0f)
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + offset_y);

        ImGui::Image(pattern_texture, sf::Vector2f(250, 250));

        ImGui::End();

        update_particle_positions(particles, turing, delta);

        // Clear window
        window.clear(sf::Color::Black);

        // Draw turing pattern hits
        for (const PatternResult& pattern : turing)
        {
            float scale_f = pattern.params.f * 1000.0f;
            float scale_k = pattern.params.k * 1000.0f;

            double justify_f = scale_f - std::fmod(scale_f, 10.0);
            double justify_k = scale_k - std::fmod(scale_k, 10.0);

            hit_rect.setPosition(justify_f, justify_k);
            window.draw(hit_rect);
        }

        if (display_particles)
        {
            // Draw particles to screen
            sf::Color particle_color = sf::Color(static_cast<uint8_t>(R), static_cast<uint8_t>(G), static_cast<uint8_t>(B));
            circle.setFillColor(particle_color);
            for (const Particle& particle : particles)
            {
                circle.setPosition(particle.pos.f * 1000.0 - 2.5, particle.pos.k * 1000.0 - 2.5);

                // Normalize directions before drawing noses for consistent length
                double dir_magnitude = std::sqrt(
                    particle.dir.f * particle.dir.f + particle.dir.k * particle.dir.k
                );
                double norm_f = particle.dir.f / dir_magnitude;
                double norm_k = particle.dir.k / dir_magnitude;

                auto nose = create_line(
                    sf::Vector2f(particle.pos.f * 1000.0, particle.pos.k * 1000.0),
                    sf::Vector2f(
                        particle.pos.f * 1000.0 + norm_f * 10,
                        particle.pos.k * 1000.0 + norm_k * 10
                    ),
                    particle_color
                );
                window.draw(circle);
                window.draw(nose);
            }
        }

        if (valid_mouse)
        {
            float hover_x = mousePos.x - std::fmod(mousePos.x, 10.0);
            float hover_y = mousePos.y - std::fmod(mousePos.y, 10.0);

            hover_rect.setPosition(hover_x, hover_y);
            window.draw(hover_rect);
        }

        if (display_axes) draw_axes(window, 0.0f, 1.0f, 0.0f, 1.0f);

        ImGui::SFML::Render(window);

        // Display
        window.display();
    }

    ImGui::SFML::Shutdown();

    return 0;
}

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
    return PatternDisplayState::SHOWING_BLACK;  // No pattern under mouse
}

void update_pattern_display(int hovered_pattern_idx, const std::vector<PatternResult>& turing, sf::Texture& pattern_texture)
{
    static int displayed_pattern_idx = PatternDisplayState::UNINITIALIZED;

    // If hovering a pattern different from what's displayed, render it
    if (hovered_pattern_idx != PatternDisplayState::SHOWING_BLACK &&
        hovered_pattern_idx != displayed_pattern_idx)
    {
        sf::Image img = image_from_pattern_data(turing[hovered_pattern_idx]);
        pattern_texture.loadFromImage(img);
        displayed_pattern_idx = hovered_pattern_idx;
    }
    // If not hovering anything and not already showing black, load black
    else if (hovered_pattern_idx == PatternDisplayState::SHOWING_BLACK &&
             displayed_pattern_idx != PatternDisplayState::SHOWING_BLACK)
    {
        static sf::Image black_image;
        static bool black_initialized = false;
        if (!black_initialized) {
            black_image.create(100, 100, sf::Color::Black);
            black_initialized = true;
        }
        pattern_texture.loadFromImage(black_image);
        displayed_pattern_idx = PatternDisplayState::SHOWING_BLACK;
    }
    // else: already displaying the correct pattern, do nothing
}

template<typename T>
bool future_is_ready(const std::future<T>& f) {
    return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}