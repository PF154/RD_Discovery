#pragma once

#include <vector>
#include <SFML/System/Time.hpp>

struct Vec4D
{
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
void scan_particle_positions(std::vector<Particle>& particles, std::vector<Vec4D>& turing);
void update_particle_positions(std::vector<Particle>& particles, std::vector<Vec4D>& wells, const sf::Time& delta);
