#ifndef NEURONLIB_BRAIN_H
#define NEURONLIB_BRAIN_H

#include "spatial.h"
#include "neuron.h"
#include "sensor.h"
#include "actuator.h"
#include "activation.h"
#include <cstdint>
#include <memory>

namespace neuronlib {

// Configurable addressing scheme
constexpr uint32_t DENDRITE_ADDRESS_BITS = 12;  // Bits 0-11 for dendrite structure
constexpr uint32_t NEURON_ADDRESS_BITS = 11;    // Configurable neuron addressing bits
constexpr uint32_t TOTAL_ADDRESS_BITS = 32;     // Total address space

// Static assertions to ensure addressing scheme is valid
static_assert(DENDRITE_ADDRESS_BITS + NEURON_ADDRESS_BITS <= TOTAL_ADDRESS_BITS, 
              "Dendrite and neuron address bits cannot exceed total address space");
static_assert(NEURON_ADDRESS_BITS > 0, "Must have at least 1 bit for neuron addressing");
static_assert(DENDRITE_ADDRESS_BITS == 12, "Dendrite addressing must be 12 bits (hardcoded in dendrite.h)");

constexpr size_t MAX_ADDRESSES = static_cast<size_t>(UINT32_MAX) + 1;
constexpr size_t ACTIVATION_ARRAY_SIZE = MAX_ADDRESSES / 8;
constexpr size_t MAX_NEURONS = 1U << NEURON_ADDRESS_BITS;  // 2^11 = 2048 neurons

struct Brain {
    SpatialGridPtr spatial_grid;
    float weights[MAX_ADDRESSES];
    uint32_t last_activations[ACTIVATION_ARRAY_SIZE];
    Neuron neurons[MAX_NEURONS];
    SensorGrid sensor_grid;
    ActuationQueue actuation_queue;
    
    Brain();
    
    Brain(const Brain&) = delete;
    Brain& operator=(const Brain&) = delete;
    Brain(Brain&&) = delete;
    Brain& operator=(Brain&&) = delete;
};

using BrainPtr = std::unique_ptr<Brain>;

} // namespace neuronlib

#endif // NEURONLIB_BRAIN_H