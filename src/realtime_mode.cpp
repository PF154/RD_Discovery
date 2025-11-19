#include "realtime_mode.h"
#include <SFML/Graphics.hpp>
#include <imgui.h>
#include <imgui-SFML.h>
#include <vector>
#include <optional>
#include <cstdint>
#include "rendering.h"

void run_real_time_sim(sf::RenderWindow& window, AppMode& mode, sf::Clock& clock, std::optional<PatternViewerState>& pattern_viewer)
{
    static int steps_per_frame = 30;

    sf::Time delta = clock.restart();
    ImGui::SFML::Update(window, delta);

    // Clear window
    window.clear(sf::Color::Black);

    // ImGui window
    ImGui::SetNextWindowPos(ImVec2(1000, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(300, 1000), ImGuiCond_Always);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize;
    ImGui::Begin("Pattern Viewer", nullptr, flags);

    ImGui::Text("Timestep: %d", pattern_viewer->sim->get_timestep());

    ImGui::SliderInt("Steps per frame", &steps_per_frame, 1, 200);

    if (ImGui::Button("Reset")) pattern_viewer->sim->reset();

    if (ImGui::Button("Return to Particle Swarm")) {
        pattern_viewer.reset();  // Calls reset on optional, not pattern sim
        mode = AppMode::PARTICLE_SWARM;
        ImGui::End();
        ImGui::SFML::Render(window);
        window.display();
        return;
    }

    ImGui::End();

    // Everyframe, progress simulation
    pattern_viewer->sim->step(steps_per_frame);

    std::vector<double> u_out, v_out;
    pattern_viewer->sim->get_state(u_out, v_out);

    sf::Image frame = std::move(
        create_image_from_simulation(
            u_out,
            v_out,
            pattern_viewer->sim->get_nx(),
            pattern_viewer->sim->get_ny()
    ));
    // Update the texture
    pattern_viewer->display_texture.loadFromImage(frame);

    // Draw simulation texture to screen
    static sf::RectangleShape pattern(sf::Vector2f(1000, 1000));
    pattern.setTexture(&pattern_viewer->display_texture);

    window.draw(pattern);

    ImGui::SFML::Render(window);
    window.display();
}

/**
 * Create an sf::Image based on simulation data.
 *
 * @param u Chemical u concentration data
 * @param v Chemical v concentration data
 * @param Nx Grid x size
 * @param Ny Grid y size
 * @return complete image of timestep
 */
sf::Image create_image_from_simulation(
    const std::vector<double>& u,
    const std::vector<double>& v,
    int Nx, int Ny
)
{
    sf::Image image;
    image.create(Nx, Ny);

    for (int i = 0; i < Nx * Ny; i++)
    {
        double pix_u = u[i];
        double pix_v = v[i];

        uint8_t r = 0;
        uint8_t g = static_cast<uint8_t>(std::clamp(pix_v * 255.0, 0.0, 255.0));
        uint8_t b = static_cast<uint8_t>(std::clamp(pix_u * 255.0, 0.0, 255.0));

        int x = i % Nx;
        int y = i / Nx;
        image.setPixel(x, y, sf::Color(r, g, b));
    }

    return image;
}