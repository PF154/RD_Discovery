#include "pattern_quadtree.h"
#include "tuning_parameters.h"
#include <cmath>
#include <algorithm>
#include <memory>
#include <iostream>

// =============================================================================
// QuadTreeNode Implementation
// =============================================================================

QuadTreeNode::QuadTreeNode(const FKExtents& bounds, int depth)
    : bounds(bounds), total_weight(0.0), depth(depth)
{
    center_of_mass = Vec4D{0.0, 0.0, 0.0, 0.0};
}

bool QuadTreeNode::is_leaf() const
{
    if (!children[0]) return true;
    return false;
}

bool QuadTreeNode::is_empty() const
{
    if (pattern_indices.size() == 0) return true;
    return false;
}

// =============================================================================
// PatternQuadTree Implementation
// =============================================================================

PatternQuadTree::PatternQuadTree(
    const FKExtents& world_bounds,
    int max_patterns_per_cell,
    double theta
)
    : m_world_bounds(world_bounds),
      m_max_patterns_per_cell(max_patterns_per_cell),
      m_theta(theta)
{
    m_root = std::make_unique<QuadTreeNode>(world_bounds);
}

void PatternQuadTree::insert(size_t pattern_index, const std::vector<PatternResult>& patterns)
{
    insert_recursive(m_root.get(), pattern_index, patterns);
}

void PatternQuadTree::insert_recursive(QuadTreeNode* node, size_t pattern_index, const std::vector<PatternResult>& patterns)
{
    if (node->is_leaf())
    {
        if (node->pattern_indices.size() < m_max_patterns_per_cell || node->depth >= MAX_DEPTH)
        {
            // Either we have room, or we've hit max depth and can't subdivide
            node->pattern_indices.push_back(pattern_index);
            return;
        }
        else
        {
            subdivide(node, patterns);
        }
    }

    // Figure out which child to recur into
    const PatternResult& pattern = patterns[pattern_index];
    int next = get_quadrant(node, pattern.params.f, pattern.params.k);

    if (next != -1) insert_recursive(node->children[next].get(), pattern_index, patterns);
    else std::cerr << "FAILED INSERT" << std::endl; // SHOULD NEVER HAPPEN
}

void PatternQuadTree::subdivide(QuadTreeNode* node, const std::vector<PatternResult>& patterns)
{
    std::vector<FKExtents> quadrant_extents;
    quadrant_extents.reserve(4);

    Vec4D current_midpoint{
        (node->bounds.max_f + node->bounds.min_f) / 2,
        (node->bounds.max_k + node->bounds.min_k) / 2,
        0.0,
        0.0
    };

    // Top left
    quadrant_extents.emplace_back(FKExtents{
        node->bounds.min_f, current_midpoint.f,
        node->bounds.min_k, current_midpoint.k
    });

    // Top right
    quadrant_extents.emplace_back(FKExtents{
        current_midpoint.f, node->bounds.max_f,
        node->bounds.min_k, current_midpoint.k
    });

    // Bottom left
    quadrant_extents.emplace_back(FKExtents{
        node->bounds.min_f, current_midpoint.f,
        current_midpoint.k, node->bounds.max_k
    });

    // Bottom right
    quadrant_extents.emplace_back(FKExtents{
        current_midpoint.f, node->bounds.max_f,
        current_midpoint.k, node->bounds.max_k
    });

    // Give node its children (with depth + 1)
    for (int i = 0; i < 4; i++)
    {
        node->children[i] = std::make_unique<QuadTreeNode>(quadrant_extents[i], node->depth + 1);
    }

    // Reassign patterns to child nodes
    for (size_t pattern_index : node->pattern_indices)
    {
        const PatternResult& pattern = patterns[pattern_index];
        int dest = get_quadrant(node, pattern.params.f, pattern.params.k);

        if (dest != -1) insert_recursive(node->children[dest].get(), pattern_index, patterns);
        else std::cerr << "FAILED TO ASSIGN PATTERN TO CHILD" << std::endl; // SHOULD NEVER HAPPEN
    }

    // Clear patterns, this is no longer a leaf
    node->pattern_indices.clear();

}

int PatternQuadTree::get_quadrant(const QuadTreeNode* node, double f, double k) const
{
    // Calculate midpoint
    double mid_f = (node->bounds.min_f + node->bounds.max_f) / 2.0;
    double mid_k = (node->bounds.min_k + node->bounds.max_k) / 2.0;

    // Determine quadrant based on position relative to midpoint
    // Use < for left/top, >= for right/bottom to handle boundary cases consistently
    if (f < mid_f) {
        if (k < mid_k) return 0;  // Top left
        else return 2;             // Bottom left
    } else {
        if (k < mid_k) return 1;  // Top right
        else return 3;             // Bottom right
    }
}

void PatternQuadTree::calculate_centers_of_mass(QuadTreeNode* node, const std::vector<PatternResult>& patterns)
{
    // Base case (leaf):
    if (node->is_leaf())
    {
        node->total_weight = node->pattern_indices.size();

        Vec4D position_sum{0.0, 0.0, 0.0, 0.0};
        for (size_t pattern_index : node->pattern_indices)
        {
            const PatternResult& pattern = patterns[pattern_index];
            position_sum.f += pattern.params.f;
            position_sum.k += pattern.params.k;
            position_sum.du += pattern.params.du;
            position_sum.dv += pattern.params.dv;
        }

        Vec4D position_avg{
            position_sum.f / node->total_weight,
            position_sum.k / node->total_weight,
            position_sum.du / node->total_weight,
            position_sum.dv / node->total_weight,
        };

        node->center_of_mass = std::move(position_avg);
    }
    else // Recursive case (internal node):
    {
        node->total_weight = 0;
        node->center_of_mass = Vec4D{0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < 4; i++)
        {
            if (!node->children[i]) continue; // SHOULD NEVER HAPPEN
            calculate_centers_of_mass(node->children[i].get(), patterns);
            double weight = node->children[i]->total_weight;
            node->total_weight += node->children[i]->total_weight;
            node->center_of_mass.f += node->children[i]->center_of_mass.f * weight;
            node->center_of_mass.k += node->children[i]->center_of_mass.k * weight;
            node->center_of_mass.du += node->children[i]->center_of_mass.du * weight;
            node->center_of_mass.dv += node->children[i]->center_of_mass.dv * weight;
        }

        if (node->total_weight > 0)
        {
            node->center_of_mass.f /= node->total_weight;
            node->center_of_mass.k /= node->total_weight;
            node->center_of_mass.du /= node->total_weight;
            node->center_of_mass.dv /= node->total_weight;
        }
    }
}

void PatternQuadTree::build(const std::vector<PatternResult>& patterns)
{
    // Clear existing tree
    this->clear();

    // Create new root node
    m_root = std::make_unique<QuadTreeNode>(m_world_bounds);

    // Insert all patterns by index
    for (size_t i = 0; i < patterns.size(); i++)
    {
        this->insert(i, patterns);
    }

    // Calculate centers of mass for entire tree
    calculate_centers_of_mass(m_root.get(), patterns);
}

Vec4D PatternQuadTree::calculate_influence(
    const Vec4D& particle_pos,
    const FKExtents& extents,
    const std::vector<PatternResult>& patterns
) const
{
    if (!m_root) return Vec4D{0.0, 0.0, 0.0, 0.0};
    return calculate_influence_recursive(m_root.get(), particle_pos, extents, patterns);
}

Vec4D PatternQuadTree::calculate_influence_recursive(
    const QuadTreeNode* node,
    const Vec4D& particle_pos,
    const FKExtents& extents,
    const std::vector<PatternResult>& patterns
) const
{
    // If node is empty, return zero influence
    if (node->is_empty()) return Vec4D{0.0, 0.0, 0.0, 0.0};

    // Calculate distance from particle to node's center of mass
    double center_f_dist = particle_pos.f - node->center_of_mass.f;
    double center_k_dist = particle_pos.k - node->center_of_mass.k;
    double center_dist = sqrt(center_f_dist * center_f_dist + center_k_dist * center_k_dist);


    Vec4D influence{0.0, 0.0, 0.0, 0.0};

    double extent_size_f = extents.max_f - extents.min_f;
    double extent_size_k = extents.max_k - extents.min_k;
    double avg_extent_size = (extent_size_f + extent_size_k) / 2.0;

    double well_strength = WELL_STRENGTH_MULTIPLIER * avg_extent_size * avg_extent_size;
    double scaled_safe_distance = MIN_SAFE_DISTANCE * avg_extent_size;  // Scale with extent size

    if (node->is_leaf())
    {
        // Sum influence of each pattern
        for (size_t pattern_index : node->pattern_indices)
        {
            const PatternResult& pattern = patterns[pattern_index];

            // Skip patterns outside current extents
            if (pattern.params.f < extents.min_f || pattern.params.f > extents.max_f ||
                pattern.params.k < extents.min_k || pattern.params.k > extents.max_k)
            {
                continue;
            }

            double df = particle_pos.f - pattern.params.f;
            double dk = particle_pos.k - pattern.params.k;
            double ddu = particle_pos.du - pattern.params.du;
            double ddv = particle_pos.dv - pattern.params.dv;
            double dist = sqrt(df * df + dk * dk + ddu * ddu + ddv * ddv);

            double safe_distance = std::max(dist, scaled_safe_distance);  // Prevent division by near-zero
            double influence_magnitude = std::min(
                MAX_INFLUENCE,
                well_strength / std::pow(safe_distance, 3)
            );
            influence.f += -df * influence_magnitude;
            influence.k += -dk * influence_magnitude;
            influence.du += -ddu * influence_magnitude;
            influence.dv += -ddv * influence_magnitude;
        }

        return influence;
    }

    else
    {
        // Calculate Barnes-Hut criterion
        double node_width = node->bounds.max_f - node->bounds.min_f;
        double theta = node_width / center_dist;

        if (theta < m_theta) // Node is far enough to be treated as single point
        {
            double df = particle_pos.f - node->center_of_mass.f;
            double dk = particle_pos.k - node->center_of_mass.k;
            double ddu = particle_pos.du - node->center_of_mass.du;
            double ddv = particle_pos.dv - node->center_of_mass.dv;
            double dist = sqrt(df * df + dk * dk + ddu * ddu + ddv * ddv);

            double safe_distance = std::max(dist, scaled_safe_distance);  // Prevent division by near-zero
            double influence_magnitude = std::min(
                MAX_INFLUENCE,
                well_strength / std::pow(safe_distance, 3)
            );
            influence.f += -df * influence_magnitude;
            influence.k += -dk * influence_magnitude;
            influence.du += -ddu * influence_magnitude;
            influence.dv += -ddv * influence_magnitude;

            return influence;
        }
        else
        {
            // Sum influence from children
            for (int i = 0; i < 4; i++)
            {
                if (node->children[i])
                {
                    Vec4D child_influence = calculate_influence_recursive(
                        node->children[i].get(),
                        particle_pos,
                        extents,
                        patterns
                    );

                    influence.f += child_influence.f;
                    influence.k += child_influence.k;
                    influence.du += child_influence.du;
                    influence.dv += child_influence.dv;
                }
            }

            return influence;
        }
    }
}

void PatternQuadTree::clear()
{
    m_root.reset();
}

size_t PatternQuadTree::count_nodes() const
{
    if (!m_root) return 0;
    return count_nodes_recursive(m_root.get());
}

size_t PatternQuadTree::count_nodes_recursive(const QuadTreeNode* node) const
{
    if (!node) return 0;  // Base case

    size_t count = 1;
    for (int i = 0; i < 4; i++)
    {
        if (node->children[i])
        {
            count += count_nodes_recursive(node->children[i].get());
        }
    }
    
    return count;
}

size_t PatternQuadTree::get_max_depth() const
{
    if (!m_root) return 0;
    return get_max_depth_recursive(m_root.get());
}

size_t PatternQuadTree::get_max_depth_recursive(const QuadTreeNode* node) const
{
    if (!node) return 0;
    
    if (node->is_leaf()) return 1;
    
    // Find maximum depth among all children
    size_t max_child_depth = 0;
    for (int i = 0; i < 4; i++)
    {
        if (node->children[i])
        {
            size_t child_depth = get_max_depth_recursive(node->children[i].get());
            max_child_depth = std::max(max_child_depth, child_depth);
        }
    }
    
    return 1 + max_child_depth;
}
