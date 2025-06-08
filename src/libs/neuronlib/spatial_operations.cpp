#include "spatial_operations.h"
#include <memory>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>

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

} // namespace neuronlib