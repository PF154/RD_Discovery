#pragma once

#include "pattern_detection.cuh"
#include <vector>
#include <string>

void export_patterns_csv(
    const std::vector<PatternResult>& results,
    const std::string& filename
);
