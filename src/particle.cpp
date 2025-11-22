#include "particle.h"
#include "async_pattern_detector.h"
#include "pattern_detection.cuh"
#include "pattern_quadtree.h"
#include "tuning_parameters.h"
#include <SFML/System/Time.hpp>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

void scan_particle_positions(
    std::vector<Particle>& particles,
    AsyncPatternDetector& detector,
    int& request_id
)
{
    // Pick batch_size random particles to test
    constexpr int batch_size = PARTICLE_BATCH_SIZE;
    if (particles.size() < batch_size) return;

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
            PARTICLE_DX,
            PARTICLE_DT
        });
    }

    detector.submit_work(std::move(param_batch), request_id++);
}

void update_particle_positions(
    std::vector<Particle>& particles, 
    std::vector<PatternResult>& wells, 
    const sf::Time& delta,
    const FKExtents& extents
)
{
    for (Particle& particle: particles)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> infl_dist(PARTICLE_INFLUENCE_MIN, PARTICLE_INFLUENCE_MAX);
        std::uniform_real_distribution<double> speed_dist(PARTICLE_SPEED_MIN, PARTICLE_SPEED_MAX);

        double extent_size_f = extents.max_f - extents.min_f;
        double extent_size_k = extents.max_k - extents.min_k;
        double avg_extent_size = (extent_size_f + extent_size_k) / 2.0;
        particle.speed = speed_dist(gen);

        double well_strength = WELL_STRENGTH_MULTIPLIER * avg_extent_size * avg_extent_size;
        constexpr double max_influence = MAX_INFLUENCE;
        constexpr double max_cumulative_influence = MAX_CUMULATIVE_INFLUENCE;
        Vec4D influence(0.0, 0.0, 0.0, 0.0);

        Vec4D random_influence(infl_dist(gen), infl_dist(gen), infl_dist(gen), infl_dist(gen));
        for (const PatternResult& well : wells)
        {
            if (well.params.f < extents.min_f || well.params.f > extents.max_f ||
                well.params.k < extents.min_k || well.params.k > extents.max_k)
            {
                continue;
            }

            double df = particle.pos.f - well.params.f;
            double dk = particle.pos.k - well.params.k;
            double ddu = particle.pos.du - well.params.du;
            double ddv = particle.pos.dv - well.params.dv;
            double distance = std::sqrt(df * df + dk * dk + ddu * ddu + ddv * ddv);

            double safe_distance = std::max(distance, MIN_SAFE_DISTANCE);  // Prevent division by near-zero
            double influence_magnitude = std::min(
                max_influence,
                well_strength / std::pow(safe_distance, 3)
            );
            influence.f = std::min(influence.f - df * influence_magnitude, max_cumulative_influence);
            influence.k = std::min(influence.k - dk * influence_magnitude, max_cumulative_influence);
            influence.du = std::min(influence.du - ddu * influence_magnitude, max_cumulative_influence);
            influence.dv = std::min(influence.dv - ddv * influence_magnitude, max_cumulative_influence);
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

        // Calculate extent ranges
        double range_f = extents.max_f - extents.min_f;
        double range_k = extents.max_k - extents.min_k;

        // Apply speed and scale by extent ranges when updating position
        // This makes speed relative to the extent size in each dimension
        particle.pos.f += particle.dir.f * particle.speed * range_f * delta.asSeconds();
        particle.pos.k += particle.dir.k * particle.speed * range_k * delta.asSeconds();

        // Particle position is periodic within extents
        particle.pos.f = extents.min_f + std::fmod(particle.pos.f - extents.min_f + range_f, range_f);
        particle.pos.k = extents.min_k + std::fmod(particle.pos.k - extents.min_k + range_k, range_k);
    }
}

void update_particle_positions_with_quadtree(
    std::vector<Particle>& particles,
    PatternQuadTree& quadtree,
    const sf::Time& delta,
    const FKExtents& extents,
    const std::vector<PatternResult>& patterns
)
{
    for (Particle& particle: particles)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> infl_dist(PARTICLE_INFLUENCE_MIN, PARTICLE_INFLUENCE_MAX);
        std::uniform_real_distribution<double> speed_dist(PARTICLE_SPEED_MIN, PARTICLE_SPEED_MAX);

        double extent_size_f = extents.max_f - extents.min_f;
        double extent_size_k = extents.max_k - extents.min_k;
        double avg_extent_size = (extent_size_f + extent_size_k) / 2.0;
        particle.speed = speed_dist(gen);

        double well_strength = WELL_STRENGTH_MULTIPLIER * avg_extent_size * avg_extent_size;
        constexpr double max_influence = MAX_INFLUENCE;
        constexpr double max_cumulative_influence = MAX_CUMULATIVE_INFLUENCE;

        Vec4D random_influence(infl_dist(gen), infl_dist(gen), infl_dist(gen), infl_dist(gen));
        Vec4D influence = quadtree.calculate_influence(particle.pos, extents, patterns);

        influence.f = std::min(influence.f, MAX_CUMULATIVE_INFLUENCE);
        influence.k = std::min(influence.k, MAX_CUMULATIVE_INFLUENCE);

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

        // Calculate extent ranges
        double range_f = extents.max_f - extents.min_f;
        double range_k = extents.max_k - extents.min_k;

        // Apply speed and scale by extent ranges when updating position
        // This makes speed relative to the extent size in each dimension
        particle.pos.f += particle.dir.f * particle.speed * range_f * delta.asSeconds();
        particle.pos.k += particle.dir.k * particle.speed * range_k * delta.asSeconds();

        // Particle position is periodic within extents
        particle.pos.f = extents.min_f + std::fmod(particle.pos.f - extents.min_f + range_f, range_f);
        particle.pos.k = extents.min_k + std::fmod(particle.pos.k - extents.min_k + range_k, range_k);
    }
}