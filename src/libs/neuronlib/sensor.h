#ifndef NEURONLIB_SENSOR_H
#define NEURONLIB_SENSOR_H

#include "geometry.h"
#include "activation.h"
#include <cstdint>
#include <vector>

namespace neuronlib {

constexpr size_t NUM_ACTIVATION_MODES = 4;
constexpr size_t MAX_DENDRITES_PER_MODE = 16;

struct ActivationMode {
    uint32_t target_dendrites[MAX_DENDRITES_PER_MODE];
    
    ActivationMode() {
        for (size_t i = 0; i < MAX_DENDRITES_PER_MODE; ++i) {
            target_dendrites[i] = 0;
        }
    }
};

struct Sensor {
    Vec3 position;
    ActivationMode modes[NUM_ACTIVATION_MODES];
    
    Sensor() : position{0, 0, 0} {}
};

constexpr size_t MAX_SENSORS = 1024;

struct SensorGrid {
    Sensor sensors[MAX_SENSORS];
    uint32_t grid_width;
    uint32_t grid_height;
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    
    SensorGrid() : grid_width(0), grid_height(0)
                 , min_x(0), min_y(0), min_z(0)
                 , max_x(0), max_y(0), max_z(0) {}
};

struct SensorActivation {
    uint32_t sensor_index;
    uint8_t mode_bitmap;  // 4 bits for 4 modes
    float value;
    
    SensorActivation() : sensor_index(0), mode_bitmap(0), value(1.0f) {}
    SensorActivation(uint32_t sensor_idx, uint8_t bitmap, float val)
        : sensor_index(sensor_idx), mode_bitmap(bitmap), value(val) {}
};

std::vector<TargetedActivation> process_sensor_activations(
    const SensorGrid& sensor_grid,
    const std::vector<SensorActivation>& activations,
    uint32_t timestamp);

void populate_sensor_grid(SensorGrid& sensor_grid,
                         uint32_t grid_width, uint32_t grid_height,
                         float min_x, float min_y, float z_plane,
                         float max_x, float max_y);

void assign_dendrites_to_sensors(SensorGrid& sensor_grid,
                                const class Brain& brain,
                                float connection_radius,
                                uint32_t random_seed = 54321);

} // namespace neuronlib

#endif // NEURONLIB_SENSOR_H