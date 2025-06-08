#include "brain.h"
#include <cstring>

namespace neuronlib {

Brain::Brain() : spatial_grid(nullptr) {
    // Initialize weights to zero
    std::memset(weights, 0, sizeof(weights));
    
    // Initialize last activations to zero
    std::memset(last_activations, 0, sizeof(last_activations));
    
    // Initialize neurons to zero
    std::memset(neurons, 0, sizeof(neurons));
}

} // namespace neuronlib