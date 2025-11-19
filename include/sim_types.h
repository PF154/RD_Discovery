#pragma once

// Pattern display state values
enum PatternDisplayState {
    UNINITIALIZED = -2,
    SHOWING_BLACK = -1
    // Values >= 0 represent pattern indices
};

enum AppMode
{
    PARTICLE_SWARM,
    REAL_TIME_PATTERN
};