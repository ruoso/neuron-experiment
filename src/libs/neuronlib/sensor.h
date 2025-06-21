#ifndef NEURONLIB_SENSOR_H
#define NEURONLIB_SENSOR_H

#include "geometry.h"
#include "activation.h"
#include <cstdint>
#include <vector>
#include <unordered_map>

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
    uint16_t sensor_tag;  // Tag identifying this sensor
    
    Sensor() : position{0, 0, 0}, sensor_tag(0) {}
};

constexpr size_t MAX_SENSORS = 1024;

struct SensorGrid {
    Sensor sensors[MAX_SENSORS];
    uint32_t grid_width;
    uint32_t grid_height;
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    std::unordered_map<uint16_t, uint32_t> tag_to_index;  // Maps sensor tags to sensor indices
    
    SensorGrid() : grid_width(0), grid_height(0)
                 , min_x(0), min_y(0), min_z(0)
                 , max_x(0), max_y(0), max_z(0) {}
};

struct SensorActivation {
    uint16_t sensor_tag;  // Tag identifying the sensor type
    uint8_t mode_bitmap;  // 4 bits for 4 modes
    float value;
    
    SensorActivation() : sensor_tag(0), mode_bitmap(0), value(1.0f) {}
    SensorActivation(uint16_t tag, uint8_t bitmap, float val)
        : sensor_tag(tag), mode_bitmap(bitmap), value(val) {}
};

std::vector<TargetedActivation> process_sensor_activations(
    const SensorGrid& sensor_grid,
    const std::vector<SensorActivation>& activations,
    uint32_t timestamp);

void populate_sensor_grid(SensorGrid& sensor_grid,
                         uint32_t grid_width, uint32_t grid_height,
                         float min_x, float min_y, float z_plane,
                         float max_x, float max_y);

void populate_sensor_grid_with_positions(SensorGrid& sensor_grid,
                                        const std::vector<struct SensorPosition>& sensor_positions);

void assign_dendrites_to_sensors(SensorGrid& sensor_grid,
                                class Brain& brain,
                                float connection_radius,
                                uint32_t random_seed = 54321);

} // namespace neuronlib

#endif // NEURONLIB_SENSOR_H