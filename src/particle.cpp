#include "particle.h"
#include "pattern_detection.cuh"
#include <random>
#include <cmath>

void scan_particle_positions(std::vector<Particle>& particles, std::vector<Vec4D>& turing)
{
    int batch_size = 25;
    std::vector<size_t> indices(particles.size());
    std::iota(indices.begin(), indices.end(), 0);

    static std::random_device rd;
    static std::mt19937 gen(rd());

    for (size_t i = 0; i < batch_size; i++) {
        std::uniform_int_distribution<size_t> dist(i, particles.size() - 1);
        size_t j = dist(gen);
        std::swap(indices[i], indices[j]);
    }

    std::vector<Particle*> batch_to_test;
    batch_to_test.reserve(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        batch_to_test.push_back(&particles[indices[i]]);
    }

    // Skip if no particles to test
    if (batch_to_test.empty()) return;

    // Assemble ParamSet vector from selected particles
    std::vector<ParamSet> param_batch;
    param_batch.reserve(batch_to_test.size());

    for (const auto* particle : batch_to_test) {
        param_batch.emplace_back(ParamSet{
            particle->pos.f,
            particle->pos.k,
            particle->pos.du,
            particle->pos.dv,
            0.25,  // dx
            1.0    // dt
        });
    }

    // Run pattern detection pipeline
    std::vector<PatternResult> results = detect_patterns_batch(param_batch);

    // Convert oscillating patterns to Vec4D and add to turing vector
    for (const auto& result : results) {
        if (result.classification == PatternType::OSCILLATING_PATTERN) {
            turing.emplace_back(
                result.params.f,
                result.params.k,
                result.params.du,
                result.params.dv
            );
        }
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
