#ifndef NEURONLIB_ACTUATOR_H
#define NEURONLIB_ACTUATOR_H

#include "geometry.h"
#include "activation.h"
#include <cstdint>
#include <vector>

namespace neuronlib {

struct ActuationEvent {
    Vec3 position;
    uint32_t timestamp;
    uint8_t actuator_tag;  // Tag identifying the actuator type
    
    ActuationEvent() : position{0, 0, 0}, timestamp(0), actuator_tag(0) {}
    ActuationEvent(const Vec3& pos, uint32_t ts, uint8_t tag = 0) 
        : position(pos), timestamp(ts), actuator_tag(tag) {}
};

void mark_actuator_neurons(class Brain& brain, float actuator_z_threshold);

void mark_actuator_neurons_with_positions(class Brain& brain, 
                                         const std::vector<struct ActuatorPosition>& actuator_positions);

} // namespace neuronlib

#endif // NEURONLIB_ACTUATOR_H