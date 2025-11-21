#include "debug_tools.h"
#include <fstream>
#include <iostream>

void export_patterns_csv(
    const std::vector<PatternResult>& results,
    const std::string& filename
)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing\n";
        return;
    }

    // Write CSV header
    file << "f,k,du,dv,classification,spatial_ratio,max_power,peak_kx,peak_ky,"
         << "peak_wave_number,temporal_change\n";

    // Write one row per result
    for (const auto& result : results) {
        file << result.params.f << ","
             << result.params.k << ","
             << result.params.du << ","
             << result.params.dv << ",";

        // Classification as string
        switch (result.classification) {
            case NO_PATTERN:          file << "NO_PATTERN,"; break;
            case TURING_PATTERN:      file << "TURING_PATTERN,"; break;
            case OSCILLATING_PATTERN: file << "OSCILLATING_PATTERN,"; break;
            default:                  file << "UNKNOWN,"; break;
        }

        file << result.spatial_ratio << ","
             << result.max_power << ","
             << result.peak_kx << ","
             << result.peak_ky << ","
             << result.peak_wave_number << ","
             << result.temporal_change << "\n";
    }

    file.close();
    std::cout << "Exported " << results.size() << " patterns to " << filename << std::endl;
}
