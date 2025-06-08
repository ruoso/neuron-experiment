#ifndef NEURONLIB_NEURON_H
#define NEURONLIB_NEURON_H

#include <cstdint>

namespace neuronlib {

constexpr size_t MAX_OUTPUT_TARGETS = 64;

struct Neuron {
    float threshold;
    uint32_t output_targets[MAX_OUTPUT_TARGETS];
};

} // namespace neuronlib

#endif // NEURONLIB_NEURON_H