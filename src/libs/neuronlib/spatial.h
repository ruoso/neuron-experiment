#ifndef NEURONLIB_SPATIAL_H
#define NEURONLIB_SPATIAL_H

#include <cstdint>
#include <variant>
#include <memory>
#include <vector>
#include "geometry.h"

namespace neuronlib {

struct SpatialBranch;
struct SpatialLeaf;
struct SpatialGrid;

using SpatialNode = std::variant<SpatialBranch, SpatialLeaf>;
using SpatialNodePtr = std::shared_ptr<const SpatialNode>;
using SpatialGridPtr = std::shared_ptr<const SpatialGrid>;

struct SpatialBranch {
    SpatialNodePtr children[8];
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    
    SpatialBranch() : min_x(0), min_y(0), min_z(0), max_x(0), max_y(0), max_z(0) {
        for (int i = 0; i < 8; ++i) {
            children[i] = nullptr;
        }
    }
};

struct SpatialItem {
    uint32_t item_address;
    Vec3 position;
    
    SpatialItem() : item_address(0), position{0, 0, 0} {}
    SpatialItem(uint32_t addr, const Vec3& pos) : item_address(addr), position(pos) {}
};

struct SpatialLeaf {
    std::vector<SpatialItem> items;
};

struct SpatialGrid {
    SpatialNodePtr root;
    uint32_t max_depth;
};

} // namespace neuronlib

#endif // NEURONLIB_SPATIAL_H