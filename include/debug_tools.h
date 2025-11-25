#pragma once

#include "pattern_detection.cuh"
#include <vector>
#include <string>

/**
 * Export PatternResult to file for manual or external analysis
 *
 * @param results Vector of PatternResult objects to be exported
 * @param filename Name of export destination
*/
void export_patterns_csv(
    const std::vector<PatternResult>& results,
    const std::string& filename
);
