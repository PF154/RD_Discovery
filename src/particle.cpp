#include "particle.h"
#include <random>
#include <cmath>

bool is_periodic(Vec4D& position)
{
    // Eventually, this will do some better computation to determine
    // if the parameters actually result in a turing pattern.
    // For now, it will just return true 0.0001 percent of the time.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> pos_dist(0.0, 1.0);

    if (pos_dist(gen) < 0.0001) return true;
    return false;
}

void scan_particle_positions(std::vector<Particle>& particles, std::vector<Vec4D>& turing)
{
    for (Particle& particle: particles)
    {
        if (is_periodic(particle.pos)) turing.push_back(particle.pos);
    }
}

void update_particle_positions(std::vector<Particle>& particles, const sf::Time& delta)
{
    for (Particle& particle: particles)
    {
        particle.pos.f += particle.dir.f * particle.speed * delta.asSeconds();
        particle.pos.k += particle.dir.k * particle.speed * delta.asSeconds();

        // Periodic boundaries
        particle.pos.f = std::fmod(particle.pos.f + 1.0, 1.0);
        particle.pos.k = std::fmod(particle.pos.k + 1.0, 1.0);
    }
}
