#pragma once

// ============================================================================
// PATTERN DETECTION PARAMETERS
// ============================================================================

// Grid size for pattern detection
constexpr int PATTERN_NX = 100;
constexpr int PATTERN_NY = 100;

// Simulation timesteps
constexpr int INITIAL_TIMESTEPS = 5000;
constexpr int STABILITY_TIMESTEPS = 2500;

// Pattern classification thresholds
// constexpr double SPATIAL_RATIO_THRESHOLD = 25.0;
// constexpr double TEMPORAL_CHANGE_THRESHOLD = 0.1;

constexpr double SPATIAL_RATIO_THRESHOLD = 500.0;
constexpr double TEMPORAL_CHANGE_THRESHOLD = 0.23;
constexpr double MIN_PEAK_POWER = 10000.0;
constexpr int MIN_WAVE_NUMBER = 3;
constexpr int MAX_WAVE_NUMBER = 25;

// CUDA kernel configuration for pattern detection
constexpr int PATTERN_THREADS_X = 16;
constexpr int PATTERN_THREADS_Y = 16;
constexpr int PATTERN_THREADS_Z = 1;
constexpr int FFT_THREADS = 256;
constexpr int POWER_ANALYSIS_THREADS = 256;

// ============================================================================
// PARTICLE SYSTEM PARAMETERS
// ============================================================================

// Number of particles in swarm
constexpr int NUM_PARTICLES = 400;

// Particle scanning
constexpr int PARTICLE_BATCH_SIZE = 25;
constexpr double PARTICLE_SCAN_INTERVAL = 0.5;  // seconds

// Particle simulation parameters (used for pattern detection)
constexpr double PARTICLE_DX = 0.25;
constexpr double PARTICLE_DT = 1.0;

// ============================================================================
// PARTICLE PHYSICS PARAMETERS
// ============================================================================

// Random influence range
constexpr double PARTICLE_INFLUENCE_MIN = -0.10;
constexpr double PARTICLE_INFLUENCE_MAX = 0.10;

// Speed distribution range
constexpr double PARTICLE_SPEED_MIN = 0.05;
constexpr double PARTICLE_SPEED_MAX = 0.15;

// Well attraction parameters
constexpr double WELL_STRENGTH_MULTIPLIER = 0.00001;
constexpr double MAX_INFLUENCE = 4.5;
constexpr double MAX_CUMULATIVE_INFLUENCE = 10.0;
constexpr double MIN_SAFE_DISTANCE = 0.01;
