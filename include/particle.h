#pragma once

#include <vector>
#include <SFML/System/Time.hpp>

#include "sim_types.h"
#include "pattern_detection.cuh"

// Need this for submitting work to the GPU manager thread
class AsyncPatternDetector;
class PatternQuadTree;

struct Vec4D
{
    Vec4D () {};
    Vec4D(double f, double k, double du, double dv)
        : f(f), k(k), du(du), dv(dv) {};
    double f = 0.0;
    double k = 0.0;
    double du = 0.0;
    double dv = 0.0;
};

struct Particle
{
    Particle(Vec4D pos, double speed, Vec4D dir)
        : pos(pos), speed(speed), dir(dir) {};
    Vec4D pos;
    double speed;
    Vec4D dir;
};

// Particle system functions
void scan_particle_positions(
    std::vector<Particle>& particles,
    AsyncPatternDetector& detector,
    int& request_id
);
void update_particle_positions(
    std::vector<Particle>& particles,
    std::vector<PatternResult>& wells,
    const sf::Time& delta,
    const FKExtents& extents
);
void update_particle_positions_with_quadtree(
    std::vector<Particle>& particles,
    PatternQuadTree& quadtree,
    const sf::Time& delta,
    const FKExtents& extents,
    const std::vector<PatternResult>& patterns
);
