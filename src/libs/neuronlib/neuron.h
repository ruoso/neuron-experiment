#ifndef NEURONLIB_NEURON_H
#define NEURONLIB_NEURON_H

#include "geometry.h"
#include <cstdint>
#include <cstddef>

namespace neuronlib {

constexpr size_t MAX_OUTPUT_TARGETS = 64;

struct Neuron {
    Vec3 position;
    Vec3 output_direction;
    float threshold;
    uint32_t output_targets[MAX_OUTPUT_TARGETS];
    bool is_actuator;
    uint8_t actuator_tag;  // Tag for actuator type (0=left_motor_activator, 1=left_motor_suppressor, etc.)
};

} // namespace neuronlib

#endif // NEURONLIB_NEURON_H