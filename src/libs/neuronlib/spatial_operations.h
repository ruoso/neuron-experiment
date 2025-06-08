#ifndef NEURONLIB_SPATIAL_OPERATIONS_H
#define NEURONLIB_SPATIAL_OPERATIONS_H

#include "spatial.h"
#include "geometry.h"
#include "flow_field.h"
#include "dendrite.h"
#include "brain.h"
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

struct DendriteTerminalPosition {
    uint32_t terminal_address;
    Vec3 position;
    
    DendriteTerminalPosition() : terminal_address(0), position{0, 0, 0} {}
    DendriteTerminalPosition(uint32_t addr, const Vec3& pos) : terminal_address(addr), position(pos) {}
};

std::vector<DendriteTerminalPosition> generate_dendrite_terminals(
    uint32_t neuron_address,
    const Vec3& neuron_position,
    const FlowField3D& flow_field,
    float cone_angle_degrees,
    float min_distance,
    float max_distance,
    uint32_t random_seed = 12345);

BrainPtr populate_neuron_grid(
    const FlowField3D& flow_field,
    float neuron_threshold = 1.0f,
    float dendrite_cone_angle = 45.0f,
    float dendrite_min_distance = 0.1f,
    float dendrite_max_distance = 0.5f,
    uint32_t random_seed = 12345);

} // namespace neuronlib

#endif // NEURONLIB_SPATIAL_OPERATIONS_H