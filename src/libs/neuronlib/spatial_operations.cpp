#include "spatial_operations.h"
#include "sensor.h"
#include <memory>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <random>

namespace neuronlib {

static int get_child_index(const Vec3& pos, float mid_x, float mid_y, float mid_z) {
    int index = 0;
    if (pos.x >= mid_x) index |= 1;
    if (pos.y >= mid_y) index |= 2;
    if (pos.z >= mid_z) index |= 4;
    return index;
}

static SpatialNodePtr insert_recursive(SpatialNodePtr node, 
                                      const std::vector<SpatialInsert>& items,
                                      float min_x, float min_y, float min_z,
                                      float max_x, float max_y, float max_z,
                                      uint32_t current_depth, uint32_t max_depth) {
    
    if (current_depth >= max_depth) {
        // Create leaf node
        SpatialLeaf leaf;
        
        // Add existing items if node exists and is a leaf
        if (node && std::holds_alternative<SpatialLeaf>(*node)) {
            const auto& existing_leaf = std::get<SpatialLeaf>(*node);
            leaf.items = existing_leaf.items;
        }
        
        // Add new items
        for (const auto& item : items) {
            leaf.items.emplace_back(item.item_address, item.position);
        }
        
        return std::make_shared<SpatialNode>(std::move(leaf));
    }
    
    // Create or copy branch node
    SpatialBranch branch;
    branch.min_x = min_x;
    branch.min_y = min_y;
    branch.min_z = min_z;
    branch.max_x = max_x;
    branch.max_y = max_y;
    branch.max_z = max_z;
    
    // Copy existing children if node exists and is a branch
    if (node && std::holds_alternative<SpatialBranch>(*node)) {
        const auto& existing_branch = std::get<SpatialBranch>(*node);
        for (int i = 0; i < 8; ++i) {
            branch.children[i] = existing_branch.children[i];
        }
    }
    
    // Calculate midpoints
    float mid_x = (min_x + max_x) * 0.5f;
    float mid_y = (min_y + max_y) * 0.5f;
    float mid_z = (min_z + max_z) * 0.5f;
    
    // Group items by child index
    std::array<std::vector<SpatialInsert>, 8> child_items;
    for (const auto& item : items) {
        int child_idx = get_child_index(item.position, mid_x, mid_y, mid_z);
        child_items[child_idx].push_back(item);
    }
    
    // Recursively insert into children that have items
    for (int i = 0; i < 8; ++i) {
        if (!child_items[i].empty()) {
            // Calculate child bounds
            float child_min_x = (i & 1) ? mid_x : min_x;
            float child_max_x = (i & 1) ? max_x : mid_x;
            float child_min_y = (i & 2) ? mid_y : min_y;
            float child_max_y = (i & 2) ? max_y : mid_y;
            float child_min_z = (i & 4) ? mid_z : min_z;
            float child_max_z = (i & 4) ? max_z : mid_z;
            
            branch.children[i] = insert_recursive(branch.children[i], child_items[i],
                                                 child_min_x, child_min_y, child_min_z,
                                                 child_max_x, child_max_y, child_max_z,
                                                 current_depth + 1, max_depth);
        }
    }
    
    return std::make_shared<SpatialNode>(std::move(branch));
}

SpatialGridPtr insert_batch(SpatialGridPtr grid, const std::vector<SpatialInsert>& items) {
    if (!grid || !grid->root) {
        throw std::invalid_argument("Grid must be non-null and have a root node");
    }
    
    if (items.empty()) {
        return grid;
    }
    
    // Get bounds from existing root
    float min_x, min_y, min_z, max_x, max_y, max_z;
    if (std::holds_alternative<SpatialBranch>(*grid->root)) {
        const auto& branch = std::get<SpatialBranch>(*grid->root);
        min_x = branch.min_x; min_y = branch.min_y; min_z = branch.min_z;
        max_x = branch.max_x; max_y = branch.max_y; max_z = branch.max_z;
    } else {
        throw std::invalid_argument("Root node must be a branch");
    }
    
    auto new_grid = std::make_shared<SpatialGrid>();
    new_grid->max_depth = grid->max_depth;
    new_grid->root = insert_recursive(grid->root, items,
                                    min_x, min_y, min_z, max_x, max_y, max_z,
                                    0, new_grid->max_depth);
    
    return new_grid;
}

static float distance(const Vec3& a, const Vec3& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

static Vec3 subtract(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

static float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static float length(const Vec3& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

static bool point_in_frustum(const Vec3& point, const Frustum3D& frustum) {
    Vec3 axis = subtract(frustum.base_center, frustum.apex);
    float height = length(axis);
    
    if (height == 0.0f) return false;
    
    // Normalize axis
    axis.x /= height;
    axis.y /= height;
    axis.z /= height;
    
    Vec3 point_to_apex = subtract(point, frustum.apex);
    float proj_length = dot(point_to_apex, axis);
    
    // Check if point is within height bounds
    if (proj_length < 0.0f || proj_length > height) {
        return false;
    }
    
    // Calculate radius at this height
    float t = proj_length / height;
    float radius_at_point = frustum.apex_radius + t * (frustum.base_radius - frustum.apex_radius);
    
    // Calculate distance from axis
    Vec3 proj_point = {
        frustum.apex.x + proj_length * axis.x,
        frustum.apex.y + proj_length * axis.y,
        frustum.apex.z + proj_length * axis.z
    };
    
    float dist_from_axis = distance(point, proj_point);
    
    return dist_from_axis <= radius_at_point;
}

static bool frustum_intersects_aabb(const Frustum3D& frustum,
                                   float min_x, float min_y, float min_z,
                                   float max_x, float max_y, float max_z) {
    // Get frustum axis and properties
    Vec3 axis = subtract(frustum.base_center, frustum.apex);
    float height = length(axis);
    
    if (height == 0.0f) return false;
    
    // Normalize axis
    axis.x /= height;
    axis.y /= height;
    axis.z /= height;
    
    // Test all 8 corners of the AABB
    Vec3 corners[8] = {
        {min_x, min_y, min_z}, {max_x, min_y, min_z},
        {min_x, max_y, min_z}, {max_x, max_y, min_z},
        {min_x, min_y, max_z}, {max_x, min_y, max_z},
        {min_x, max_y, max_z}, {max_x, max_y, max_z}
    };
    
    // Check if any corner is inside the frustum
    for (int i = 0; i < 8; ++i) {
        if (point_in_frustum(corners[i], frustum)) {
            return true;
        }
    }
    
    // Additional test: check if frustum axis intersects the AABB
    // and if the intersection point is within the frustum bounds
    
    // Find min/max projection of AABB onto frustum axis
    float min_proj = std::numeric_limits<float>::max();
    float max_proj = std::numeric_limits<float>::lowest();
    
    for (int i = 0; i < 8; ++i) {
        Vec3 corner_to_apex = subtract(corners[i], frustum.apex);
        float proj = dot(corner_to_apex, axis);
        min_proj = std::min(min_proj, proj);
        max_proj = std::max(max_proj, proj);
    }
    
    // Check if AABB projection overlaps with frustum height
    if (max_proj < 0.0f || min_proj > height) {
        return false;
    }
    
    // Check if frustum could pass through the AABB
    float clipped_min = std::max(0.0f, min_proj);
    float clipped_max = std::min(height, max_proj);
    
    for (float t_norm = clipped_min / height; t_norm <= clipped_max / height; t_norm += 0.1f) {
        float radius_at_t = frustum.apex_radius + t_norm * (frustum.base_radius - frustum.apex_radius);
        
        Vec3 center_at_t = {
            frustum.apex.x + t_norm * height * axis.x,
            frustum.apex.y + t_norm * height * axis.y,
            frustum.apex.z + t_norm * height * axis.z
        };
        
        // Check if circle at this height intersects AABB
        if (center_at_t.x + radius_at_t >= min_x && center_at_t.x - radius_at_t <= max_x &&
            center_at_t.y + radius_at_t >= min_y && center_at_t.y - radius_at_t <= max_y &&
            center_at_t.z + radius_at_t >= min_z && center_at_t.z - radius_at_t <= max_z) {
            return true;
        }
    }
    
    return false;
}

static void search_frustum_recursive(SpatialNodePtr node,
                                    const Frustum3D& frustum,
                                    std::vector<FrustumSearchResult>& results) {
    if (!node) return;
    
    if (std::holds_alternative<SpatialLeaf>(*node)) {
        const auto& leaf = std::get<SpatialLeaf>(*node);
        for (const auto& item : leaf.items) {
            if (point_in_frustum(item.position, frustum)) {
                float dist = distance(item.position, frustum.apex);
                results.emplace_back(item.item_address, dist);
            }
        }
    } else if (std::holds_alternative<SpatialBranch>(*node)) {
        const auto& branch = std::get<SpatialBranch>(*node);
        
        // Early exit if branch doesn't intersect with frustum
        if (!frustum_intersects_aabb(frustum, 
                                   branch.min_x, branch.min_y, branch.min_z,
                                   branch.max_x, branch.max_y, branch.max_z)) {
            return;
        }
        
        // Recursively check children
        for (int i = 0; i < 8; ++i) {
            search_frustum_recursive(branch.children[i], frustum, results);
        }
    }
}

std::vector<FrustumSearchResult> search_frustum(SpatialGridPtr grid, const Frustum3D& frustum) {
    std::vector<FrustumSearchResult> results;
    
    if (!grid || !grid->root) {
        return results;
    }
    
    search_frustum_recursive(grid->root, frustum, results);
    
    // Sort by distance to apex
    std::sort(results.begin(), results.end(), 
              [](const FrustumSearchResult& a, const FrustumSearchResult& b) {
                  return a.distance_to_apex < b.distance_to_apex;
              });
    
    return results;
}

static Vec3 normalize_vector(const Vec3& v) {
    float magnitude = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (magnitude == 0.0f) {
        return {0.0f, 0.0f, 1.0f};  // default direction
    }
    return {v.x / magnitude, v.y / magnitude, v.z / magnitude};
}

static Vec3 get_perpendicular_vector(const Vec3& v) {
    // Find a vector perpendicular to v
    if (std::abs(v.x) < 0.9f) {
        return normalize_vector({1.0f, 0.0f, 0.0f});
    } else {
        return normalize_vector({0.0f, 1.0f, 0.0f});
    }
}

static Vec3 rotate_around_axis(const Vec3& point, const Vec3& axis, float angle_radians) {
    // Rodrigues' rotation formula
    float cos_angle = std::cos(angle_radians);
    float sin_angle = std::sin(angle_radians);
    
    Vec3 cross_product = {
        axis.y * point.z - axis.z * point.y,
        axis.z * point.x - axis.x * point.z,
        axis.x * point.y - axis.y * point.x
    };
    
    float dot_product = axis.x * point.x + axis.y * point.y + axis.z * point.z;
    
    return {
        point.x * cos_angle + cross_product.x * sin_angle + axis.x * dot_product * (1.0f - cos_angle),
        point.y * cos_angle + cross_product.y * sin_angle + axis.y * dot_product * (1.0f - cos_angle),
        point.z * cos_angle + cross_product.z * sin_angle + axis.z * dot_product * (1.0f - cos_angle)
    };
}

std::vector<DendriteTerminalPosition> generate_dendrite_terminals(
    uint32_t neuron_address,
    const Vec3& neuron_position,
    const FlowField3D& flow_field,
    float cone_angle_degrees,
    float min_distance,
    float max_distance,
    uint32_t random_seed) {
    
    std::vector<DendriteTerminalPosition> terminals;
    
    // Get flow direction at neuron position and invert for dendrites (inputs)
    Vec3 flow_direction = normalize_vector(evaluate_flow_field(flow_field, neuron_position));
    Vec3 dendrite_direction = {-flow_direction.x, -flow_direction.y, -flow_direction.z};
    
    // Set up random number generator
    std::mt19937 rng(random_seed);
    std::uniform_real_distribution<float> uniform_01(0.0f, 1.0f);
    std::uniform_real_distribution<float> distance_dist(min_distance, max_distance);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
    
    float cone_angle_radians = cone_angle_degrees * M_PI / 180.0f;
    
    // Get perpendicular vectors to create a coordinate system
    Vec3 perp1 = get_perpendicular_vector(dendrite_direction);
    Vec3 perp2 = {
        dendrite_direction.y * perp1.z - dendrite_direction.z * perp1.y,
        dendrite_direction.z * perp1.x - dendrite_direction.x * perp1.z,
        dendrite_direction.x * perp1.y - dendrite_direction.y * perp1.x
    };
    perp2 = normalize_vector(perp2);
    
    // Generate terminals for all possible addresses (8 branches * 8 sub-branches * 8 terminals)
    for (uint32_t branch_l3 = 1; branch_l3 <= 7; ++branch_l3) {
        for (uint32_t branch_l2 = 1; branch_l2 <= 7; ++branch_l2) {
            for (uint32_t branch_l1 = 1; branch_l1 <= 7; ++branch_l1) {
                for (uint32_t terminal_id = 1; terminal_id <= 7; ++terminal_id) {
                    // Create terminal address
                    uint32_t terminal_address = neuron_address | 
                                              (branch_l3 << 9) | 
                                              (branch_l2 << 6) | 
                                              (branch_l1 << 3) | 
                                              terminal_id;
                    
                    // Generate random position within cone
                    float distance = distance_dist(rng);
                    float phi = angle_dist(rng);  // azimuthal angle
                    
                    // Random theta within cone (using uniform distribution on cone surface)
                    float cos_theta_max = std::cos(cone_angle_radians);
                    float cos_theta = cos_theta_max + uniform_01(rng) * (1.0f - cos_theta_max);
                    float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);
                    
                    // Convert to Cartesian coordinates in cone space
                    float x = sin_theta * std::cos(phi);
                    float y = sin_theta * std::sin(phi);
                    float z = cos_theta;
                    
                    // Transform to world coordinates using dendrite direction as cone axis
                    Vec3 cone_point = {
                        x * perp1.x + y * perp2.x + z * dendrite_direction.x,
                        x * perp1.y + y * perp2.y + z * dendrite_direction.y,
                        x * perp1.z + y * perp2.z + z * dendrite_direction.z
                    };
                    
                    // Scale by distance and translate to neuron position
                    Vec3 terminal_position = {
                        neuron_position.x + cone_point.x * distance,
                        neuron_position.y + cone_point.y * distance,
                        neuron_position.z + cone_point.z * distance
                    };
                    
                    terminals.emplace_back(terminal_address, terminal_position);
                }
            }
        }
    }
    
    return terminals;
}

BrainPtr populate_neuron_grid(
    const FlowField3D& flow_field,
    float neuron_threshold,
    float dendrite_cone_angle,
    float dendrite_min_distance,
    float dendrite_max_distance,
    uint32_t sensor_grid_width,
    uint32_t sensor_grid_height,
    float sensor_connection_radius,
    uint32_t random_seed) {
    
    // Create brain
    auto brain = std::make_unique<Brain>();
    
    // Calculate 3D grid dimensions - try to make it as cubic as possible
    uint32_t cube_root = static_cast<uint32_t>(std::cbrt(MAX_NEURONS));
    uint32_t grid_x = cube_root;
    uint32_t grid_y = cube_root;
    uint32_t grid_z = MAX_NEURONS / (grid_x * grid_y);
    
    // Adjust to use all available addresses
    while (grid_x * grid_y * grid_z < MAX_NEURONS && grid_z < cube_root + 10) {
        grid_z++;
    }
    
    // Calculate spacing between neurons
    float spacing_x = (flow_field.max_x - flow_field.min_x) / (grid_x - 1);
    float spacing_y = (flow_field.max_y - flow_field.min_y) / (grid_y - 1);
    float spacing_z = (flow_field.max_z - flow_field.min_z) / (grid_z - 1);
    
    // Create initial empty spatial grid with root branch
    SpatialBranch root_branch;
    root_branch.min_x = flow_field.min_x;
    root_branch.min_y = flow_field.min_y;
    root_branch.min_z = flow_field.min_z;
    root_branch.max_x = flow_field.max_x;
    root_branch.max_y = flow_field.max_y;
    root_branch.max_z = flow_field.max_z;
    
    auto root_node = std::make_shared<SpatialNode>(std::move(root_branch));
    auto spatial_grid = std::make_shared<SpatialGrid>();
    spatial_grid->root = root_node;
    spatial_grid->max_depth = 8;
    
    // Generate all spatial items (neurons + dendrite terminals)
    std::vector<SpatialInsert> all_items;
    
    uint32_t neuron_count = 0;
    for (uint32_t z = 0; z < grid_z && neuron_count < MAX_NEURONS; ++z) {
        for (uint32_t y = 0; y < grid_y && neuron_count < MAX_NEURONS; ++y) {
            for (uint32_t x = 0; x < grid_x && neuron_count < MAX_NEURONS; ++x) {
                // Calculate neuron position
                Vec3 neuron_position = {
                    flow_field.min_x + x * spacing_x,
                    flow_field.min_y + y * spacing_y,
                    flow_field.min_z + z * spacing_z
                };
                
                // Create neuron address (shifted left by 12 bits to leave dendrite space)
                uint32_t neuron_address = neuron_count << 12;
                
                // Initialize neuron in brain
                brain->neurons[neuron_count].position = neuron_position;
                brain->neurons[neuron_count].output_direction = normalize_vector(evaluate_flow_field(flow_field, neuron_position));
                brain->neurons[neuron_count].threshold = neuron_threshold;
                // output_targets array is already zero-initialized by memset
                
                // Add neuron to spatial items
                all_items.emplace_back(neuron_address, neuron_position);
                
                // Generate dendrite terminals for this neuron
                auto terminals = generate_dendrite_terminals(
                    neuron_address, neuron_position, flow_field,
                    dendrite_cone_angle, dendrite_min_distance, dendrite_max_distance,
                    random_seed + neuron_count);
                
                // Add all terminals to spatial items
                for (const auto& terminal : terminals) {
                    all_items.emplace_back(terminal.terminal_address, terminal.position);
                }
                
                neuron_count++;
            }
        }
    }
    
    // Insert all items into the spatial grid
    brain->spatial_grid = insert_batch(spatial_grid, all_items);
    
    // Populate sensor grid at the bottom Z plane of the flow field
    populate_sensor_grid(brain->sensor_grid,
                        sensor_grid_width, sensor_grid_height,
                        flow_field.min_x, flow_field.min_y, flow_field.min_z,
                        flow_field.max_x, flow_field.max_y);
    
    // Assign dendrites to sensor modes
    assign_dendrites_to_sensors(brain->sensor_grid, *brain, 
                               sensor_connection_radius, random_seed + 999999);
    
    return brain;
}

} // namespace neuronlib