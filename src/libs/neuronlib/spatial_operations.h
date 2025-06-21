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

struct BoundingBoxSearchResult {
    uint32_t item_address;
    Vec3 position;
    
    BoundingBoxSearchResult() : item_address(0), position{0, 0, 0} {}
    BoundingBoxSearchResult(uint32_t addr, const Vec3& pos) : item_address(addr), position(pos) {}
};

std::vector<BoundingBoxSearchResult> search_bounding_box(SpatialGridPtr grid,
                                                        float min_x, float min_y, float min_z,
                                                        float max_x, float max_y, float max_z);

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
    uint32_t sensor_grid_width = 32,
    uint32_t sensor_grid_height = 32,
    float sensor_connection_radius = 0.3f,
    float actuator_z_threshold = 0.1f,
    uint32_t random_seed = 12345);

struct SensorPosition {
    Vec3 position;
    uint16_t sensor_tag;
    
    SensorPosition() : position{0, 0, 0}, sensor_tag(0) {}
    SensorPosition(const Vec3& pos, uint16_t tag) : position(pos), sensor_tag(tag) {}
};

struct ActuatorPosition {
    Vec3 position;
    uint8_t tag;  // e.g., 0=left_motor_activator, 1=left_motor_suppressor, 2=right_motor_activator, 3=right_motor_suppressor
    
    ActuatorPosition() : position{0, 0, 0}, tag(0) {}
    ActuatorPosition(const Vec3& pos, uint8_t t) : position(pos), tag(t) {}
};

// Helper functions for brain initialization
void initialize_brain_weights(Brain& brain, uint32_t random_seed);
void populate_neurons_and_dendrites(Brain& brain, const FlowField3D& flow_field,
                                   float dendrite_cone_angle, float dendrite_min_distance,
                                   float dendrite_max_distance, uint32_t random_seed);

BrainPtr populate_neuron_grid_with_layout(
    const FlowField3D& flow_field,
    const std::vector<SensorPosition>& sensor_positions,
    const std::vector<ActuatorPosition>& actuator_positions,
    float neuron_threshold = 1.0f,
    float dendrite_cone_angle = 45.0f,
    float dendrite_min_distance = 0.1f,
    float dendrite_max_distance = 0.5f,
    float sensor_connection_radius = 0.3f,
    uint32_t random_seed = 12345);

} // namespace neuronlib

#endif // NEURONLIB_SPATIAL_OPERATIONS_H