#include "particle_swarm_mode.h"
#include "tuning_parameters.h"
#include "debug_tools.h"
#include <SFML/Graphics.hpp>
#include <imgui.h>
#include <imgui-SFML.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include "rendering.h"
#include "utilities.h"

void run_particle_swarm(
    sf::RenderWindow& window,
    std::vector<PatternResult>& turing,
    AsyncPatternDetector& detector,
    std::vector<Particle>& particles,
    sf::Clock& clock,
    const FKExtents& extents,
    const SelectionState& selection_state,
    bool& reset_extents
)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());

    static sf::Texture pattern_texture;
    static int next_request_id = 0;

    // Particle visualisation geometry
    static sf::CircleShape circle(2.5);
    static sf::RectangleShape t_hit_rect(sf::Vector2f(10, 10));
    t_hit_rect.setFillColor(sf::Color::Green);
    static sf::RectangleShape osc_hit_rect(sf::Vector2f(10, 10));
    osc_hit_rect.setFillColor(sf::Color(153, 153, 153));
    static sf::RectangleShape no_hit_rect(sf::Vector2f(10, 10));
    no_hit_rect.setFillColor(sf::Color::Red);

    // We want a visualization of where the user's cursor is
    static sf::RectangleShape hover_rect(sf::Vector2f(10, 10));
    hover_rect.setFillColor(sf::Color::Transparent);
    hover_rect.setOutlineColor(sf::Color(3, 252, 236));
    hover_rect.setOutlineThickness(-1.0f);

    // Extent selection geometry
    static sf::RectangleShape select_rect(sf::Vector2f(1, 1));
    select_rect.setFillColor(sf::Color::Transparent);
    select_rect.setOutlineColor(sf::Color(245, 66, 224));
    select_rect.setOutlineThickness(-1.0f);

    static float R = 100.0f;
    static float G = 200.0f;
    static float B = 100.0f;

    static bool display_particles = true;
    static bool display_axes = true;

    static float du = 0.16;
    static float dv = 0.08;

    static bool extents_modified = false;

    // Control where particles can go
    static FKExtents particle_extents = extents;
    static bool constrain_particles = false;

    int hovered_pattern_idx = find_pattern_under_mouse(sf::Mouse::getPosition(window), turing, extents);
    update_pattern_display(hovered_pattern_idx, turing, pattern_texture);

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
                if (pattern.classification != PatternType::NO_PATTERN)
                {
                    turing.emplace_back(pattern);
                }
            }
        }
    }

    // Submit work at regular intervals
    static sf::Clock detection_timer;
    if (detection_timer.getElapsedTime().asSeconds() > PARTICLE_SCAN_INTERVAL) {
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

    if (ImGui::Button("Constrain particle movement to exetents"))
    {
        // Redo particle positioning
        std::uniform_real_distribution<double> f_dist(extents.min_f, extents.max_f);
        std::uniform_real_distribution<double> k_dist(extents.min_k, extents.max_k);

        for (Particle& particle : particles)
            particle.pos = Vec4D{f_dist(gen), k_dist(gen), particle.pos.du, particle.pos. dv};

        particle_extents = extents;
    }

    if (extents_modified)
    {
        if (ImGui::Button("Reset Extents"))
        {
            extents_modified = false;
            reset_extents = true;
        }
    }

    // Debug/Analysis Tools
    ImGui::Separator();
    ImGui::Text("Debug Tools");
    ImGui::Text("Patterns detected: %zu", turing.size());

    if (ImGui::Button("Export Patterns to CSV"))
    {
        export_patterns_csv(turing, "detected_patterns.csv");
    }

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

    update_particle_positions(particles, turing, delta, particle_extents);

    // Clear window
    window.clear(sf::Color::Black);

    // Draw turing pattern hits
    for (const PatternResult& pattern : turing)
    {
        // Transform from parameter space to screen space using extents
        sf::Vector2f screen_pos = param_to_screen(pattern.params.f, pattern.params.k, extents);

        double justify_f = screen_pos.x - std::fmod(screen_pos.x, 10.0);
        double justify_k = screen_pos.y - std::fmod(screen_pos.y, 10.0);

        if (pattern.classification == PatternType::TURING_PATTERN)
        {
            t_hit_rect.setPosition(justify_f, justify_k);
            window.draw(t_hit_rect);
        }
        else if (pattern.classification == PatternType::OSCILLATING_PATTERN)
        {
            osc_hit_rect.setPosition(justify_f, justify_k);
            window.draw(osc_hit_rect);
        }
        else // SHOULD NEVER HAPPEN
        {
            no_hit_rect.setPosition(justify_f, justify_k);
            window.draw(no_hit_rect);
        }
    }

    if (display_particles)
    {
        // Draw particles to screen
        sf::Color particle_color = sf::Color(static_cast<uint8_t>(R), static_cast<uint8_t>(G), static_cast<uint8_t>(B));
        circle.setFillColor(particle_color);
        for (const Particle& particle : particles)
        {
            // Transform particle position from parameter space to screen space using extents
            sf::Vector2f screen_pos = param_to_screen(particle.pos.f, particle.pos.k, extents);

            circle.setPosition(screen_pos.x - 2.5, screen_pos.y - 2.5);

            // Normalize directions before drawing noses for consistent length
            double dir_magnitude = std::sqrt(
                particle.dir.f * particle.dir.f + particle.dir.k * particle.dir.k
            );
            double norm_dir_f = particle.dir.f / dir_magnitude;
            double norm_dir_k = particle.dir.k / dir_magnitude;

            auto nose = create_line(
                screen_pos,
                sf::Vector2f(
                    screen_pos.x + norm_dir_f * 10,
                    screen_pos.y + norm_dir_k * 10
                ),
                particle_color
            );
            window.draw(circle);
            window.draw(nose);
        }
    }


    if (selection_state.is_selecting)
    {
        extents_modified = true;

        sf::Vector2f param_coords = screen_to_param(mousePos.x, mousePos.y, extents);

        float pos_x = selection_state.current_extents.min_f;
        float pos_y = selection_state.current_extents.min_k;

        sf::Vector2f screen_start = param_to_screen(pos_x, pos_y, extents);

        select_rect.setPosition(screen_start);
        select_rect.setSize(sf::Vector2f(
            mousePos.x - screen_start.x, 
            mousePos.y - screen_start.y
        ));

        window.draw(select_rect);
    }
    else if (valid_mouse && !selection_state.is_selecting)
    {
        float hover_x = mousePos.x - std::fmod(mousePos.x, 10.0);
        float hover_y = mousePos.y - std::fmod(mousePos.y, 10.0);

        hover_rect.setPosition(hover_x, hover_y);
        window.draw(hover_rect);
    }

    if (display_axes) draw_axes(window, extents.min_f, extents.max_f, extents.min_k, extents.max_k);

    ImGui::SFML::Render(window);

    // Display
    window.display();
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