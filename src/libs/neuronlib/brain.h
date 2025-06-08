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

constexpr size_t MAX_ADDRESSES = static_cast<size_t>(UINT32_MAX) + 1;
constexpr size_t ACTIVATION_ARRAY_SIZE = MAX_ADDRESSES / 8;
constexpr size_t MAX_NEURONS = 1U << 20;  // 2^20 neurons (20 bits for neuron addressing)

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