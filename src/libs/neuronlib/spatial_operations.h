#ifndef NEURONLIB_SPATIAL_OPERATIONS_H
#define NEURONLIB_SPATIAL_OPERATIONS_H

#include "spatial.h"
#include "geometry.h"
#include <vector>

namespace neuronlib {

struct SpatialInsert {
    uint32_t item_address;
    Vec3 position;
};

struct FrustumSearchResult {
    uint32_t item_address;
    float distance_to_apex;
    
    FrustumSearchResult() : item_address(0), distance_to_apex(0.0f) {}
    FrustumSearchResult(uint32_t addr, float dist) : item_address(addr), distance_to_apex(dist) {}
};

SpatialGridPtr insert_batch(SpatialGridPtr grid, const std::vector<SpatialInsert>& items);

std::vector<FrustumSearchResult> search_frustum(SpatialGridPtr grid, const Frustum3D& frustum);

} // namespace neuronlib

#endif // NEURONLIB_SPATIAL_OPERATIONS_H