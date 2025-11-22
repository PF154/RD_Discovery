#pragma once

#include "sim_types.h"
#include "particle.h"
#include "pattern_detection.cuh"
#include <vector>
#include <memory>

/**
 * QuadTreeNode represents a rectangular cell in the f-k parameter space.
 * Each node either contains patterns (leaf) or has 4 children (internal node).
 */
struct QuadTreeNode {
    // Spatial bounds of this node
    FKExtents bounds;

    // For Barnes-Hut approximation
    Vec4D center_of_mass;    // Weighted average position of all patterns in this subtree
    double total_weight;      // Total number of patterns in this subtree

    // Leaf node data
    std::vector<size_t> pattern_indices;  // Indices into the pattern vector

    // Internal node data
    std::unique_ptr<QuadTreeNode> children[4];

    int depth;

    QuadTreeNode(const FKExtents& bounds, int depth = 0);

    bool is_leaf() const;
    bool is_empty() const;
};

/**
 * PatternQuadTree implements Barnes-Hut algorithm for fast particle-pattern
 * force calculations.
 */
class PatternQuadTree {
private:
    std::unique_ptr<QuadTreeNode> m_root;
    FKExtents m_world_bounds;

    // Configuration parameters
    int m_max_patterns_per_cell;
    double m_theta;
    static constexpr int MAX_DEPTH = 20;  // Prevent infinite recursion

    // Private helper methods
    void insert_recursive(QuadTreeNode* node, size_t pattern_index, const std::vector<PatternResult>& patterns);
    void subdivide(QuadTreeNode* node, const std::vector<PatternResult>& patterns);
    int get_quadrant(const QuadTreeNode* node, double f, double k) const;
    void calculate_centers_of_mass(QuadTreeNode* node, const std::vector<PatternResult>& patterns);
    Vec4D calculate_influence_recursive(
        const QuadTreeNode* node,
        const Vec4D& particle_pos,
        const FKExtents& extents,
        const std::vector<PatternResult>& patterns
    ) const;
    size_t count_nodes_recursive(const QuadTreeNode* node) const;
    size_t get_max_depth_recursive(const QuadTreeNode* node) const;

public:
    /**
     * Constructor
     * @param world_bounds The f-k extents that define the root cell
     * @param max_patterns Maximum patterns per leaf cell before subdivision
     * @param theta Barnes-Hut approximation threshold
     */
    PatternQuadTree(
        const FKExtents& world_bounds,
        int max_patterns_per_cell = 8,
        double theta = 0.5
    );

    /**
     * Insert a pattern into the tree by index
     */
    void insert(size_t pattern_index, const std::vector<PatternResult>& patterns);

    /**
     * Build the tree from a vector of patterns (clears existing tree)
     */
    void build(const std::vector<PatternResult>& patterns);

    /**
     * Calculate the gravitational influence on a particle from all patterns
     * Uses Barnes-Hut approximation for distant groups
     *
     * @param particle_pos Current position of the particle
     * @param extents Current world extents (for scaling)
     * @param patterns The pattern vector to look up indices from
     * @return Influence vector in 4D space (f, k, du, dv)
     */
    Vec4D calculate_influence(
        const Vec4D& particle_pos,
        const FKExtents& extents,
        const std::vector<PatternResult>& patterns
    ) const;

    /**
     * Clear the tree
     */
    void clear();

    // Debugging/statistics
    size_t count_nodes() const;
    size_t get_max_depth() const;
};
