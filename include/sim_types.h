#pragma once

// Pattern display state values
enum PatternDisplayState 
{
    UNINITIALIZED = -2,
    SHOWING_BLACK = -1
    // Values >= 0 represent pattern indices
};

enum AppMode
{
    PARTICLE_SWARM,
    REAL_TIME_PATTERN
};

// Represents the extents of the simulation window
struct FKExtents
{
    double min_f;
    double min_k;
    double max_f;
    double max_k;
};

// Passed from main to paricle swarm to determine if selection window is drawn
struct SelectionState
{
    bool is_selecting = false;
    FKExtents current_extents;
};