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
    
    ActuationEvent() : position{0, 0, 0}, timestamp(0) {}
    ActuationEvent(const Vec3& pos, uint32_t ts) : position(pos), timestamp(ts) {}
};

void mark_actuator_neurons(class Brain& brain, float actuator_z_threshold);

} // namespace neuronlib

#endif // NEURONLIB_ACTUATOR_H