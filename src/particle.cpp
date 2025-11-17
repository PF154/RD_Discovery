#include "particle.h"
#include <random>
#include <cmath>

bool is_periodic(Vec4D& position)
{
    // Eventually, this will do some better computation to determine
    // if the parameters actually result in a turing pattern.
    // For now, it will just return true 0.000001 percent of the time.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> pos_dist(0.0, 1.0);

    if (pos_dist(gen) < 0.00001) return true;
    return false;
}

void scan_particle_positions(std::vector<Particle>& particles, std::vector<Vec4D>& turing)
{
    for (Particle& particle: particles)
    {
        if (is_periodic(particle.pos)) turing.push_back(particle.pos);
    }
}

void update_particle_positions(
    std::vector<Particle>& particles, 
    std::vector<Vec4D>& wells, 
    const sf::Time& delta
)
{
    for (Particle& particle: particles)
    {
        double well_strength = 0.001;
        double max_influence = 7.0;
        Vec4D influence(0.0, 0.0, 0.0, 0.0);

        // We want some randomness in particle direction to avoid getting stuck
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-0.05, 0.05);
        Vec4D random_influence(dist(gen), dist(gen), dist(gen), dist(gen));
        for (Vec4D well : wells)
        {
            // Compute Euclidean distance between particle and well (4D)
            double df = particle.pos.f - well.f;
            double dk = particle.pos.k - well.k;
            double ddu = particle.pos.du - well.du;
            double ddv = particle.pos.dv - well.dv;
            double distance = std::sqrt(df * df + dk * dk + ddu * ddu + ddv * ddv);
            
            double safe_distance = std::max(distance, 0.01);  // Prevent division by near-zero
            double influence_magnitude = std::min(
                max_influence,
                well_strength / std::pow(safe_distance, 2)
            );
            influence.f += -df * influence_magnitude;
            influence.k += -dk * influence_magnitude;
            influence.du += -ddu * influence_magnitude;
            influence.dv += -ddv * influence_magnitude;
        }

        particle.dir.f += influence.f + random_influence.f;
        particle.dir.k += influence.k + random_influence.k;

        // Normalize the direction vector to maintain unit length
        double dir_magnitude = std::sqrt(
            particle.dir.f * particle.dir.f +
            particle.dir.k * particle.dir.k +
            particle.dir.du * particle.dir.du +
            particle.dir.dv * particle.dir.dv
        );

        if (dir_magnitude > 0.0) {
            particle.dir.f /= dir_magnitude;
            particle.dir.k /= dir_magnitude;
            particle.dir.du /= dir_magnitude;
            particle.dir.dv /= dir_magnitude;
        }

        particle.pos.f += particle.dir.f * particle.speed * delta.asSeconds();
        particle.pos.k += particle.dir.k * particle.speed * delta.asSeconds();

        // Periodic boundaries
        particle.pos.f = std::fmod(particle.pos.f + 1.0, 1.0);
        particle.pos.k = std::fmod(particle.pos.k + 1.0, 1.0);
    }
}
