#pragma once

#include <vector>
#include <SFML/System/Time.hpp>

#include "sim_types.h"
#include "pattern_detection.cuh"

// Need this for submitting work to the GPU manager thread
class AsyncPatternDetector;
class PatternQuadTree;

// Represents a position in the 4D parameter space of the Gray-Scott model
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

// Represents a particle in the particle swarm
struct Particle
{
    Particle(Vec4D pos, double speed, Vec4D dir)
        : pos(pos), speed(speed), dir(dir) {};
    Vec4D pos;
    double speed;
    Vec4D dir;
};

/**
 * Searches particle positions to see if any are over a Turing pattern
 * 
 * @param paritlces Vector of Particle objects to scan the positions of
 * @param detector AsyncPatternDetector to delegate pattern detection work to GPU
 * @param request_id Request ID, can probably be removed
 */
void scan_particle_positions(
    std::vector<Particle>& particles,
    AsyncPatternDetector& detector,
    int& request_id
);


/**
 * Update particle positions based on a vector of "wells" that particles are pulled toward
 * 
 * Runs every frame
 * 
 * @param particles Vector of Particle objects to be updated
 * @param wells Vector of discovered patterns that are treated as wells of attraction for particles
 * @param delta time since last frame
 * @param extents Extents of the particle bounds
 */
void update_particle_positions(
    std::vector<Particle>& particles,
    std::vector<PatternResult>& wells,
    const sf::Time& delta,
    const FKExtents& extents
);

/**
 * Update particle positions based on a quadtree of positions that particles are pulled toward
 * 
 * This is FAR more efficient than the vector version, and allows for smooth simulation even with
 * a large number of discovered patterns.
 * Runs every frame
 * 
 * @param particles Vector of Particle objects to be updated
 * @param quadree PatternQuadTree object that represent the tree of discovered patterns
 * @param delta time since last frame
 * @param extents Extents of the particle bounds
 * @param PatternResult Soon to be deprecated
 */
void update_particle_positions_with_quadtree(
    std::vector<Particle>& particles,
    PatternQuadTree& quadtree,
    const sf::Time& delta,
    const FKExtents& extents,
    const std::vector<PatternResult>& patterns
);
