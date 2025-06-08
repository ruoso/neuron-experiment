#include "actuator.h"
#include "brain.h"
#include "spatial_operations.h"
#include "dendrite.h"
#include <algorithm>
#include <limits>

namespace neuronlib {

void mark_actuator_neurons(Brain& brain, float actuator_z_threshold) {
    if (!brain.spatial_grid || !brain.spatial_grid->root) {
        return;
    }
    
    // First, reset all neurons to not be actuators
    for (size_t i = 0; i < MAX_NEURONS; ++i) {
        brain.neurons[i].is_actuator = false;
    }
    
    // Get the bounding box of the entire spatial grid
    if (!std::holds_alternative<SpatialBranch>(*brain.spatial_grid->root)) {
        return;
    }
    
    const auto& root_branch = std::get<SpatialBranch>(*brain.spatial_grid->root);
    float min_x = root_branch.min_x;
    float min_y = root_branch.min_y;
    float min_z = root_branch.max_z - actuator_z_threshold;  // Search near max Z
    float max_x = root_branch.max_x;
    float max_y = root_branch.max_y;
    float max_z = root_branch.max_z;
    
    // Search for all items in the actuator zone
    auto search_results = search_bounding_box(brain.spatial_grid, 
                                             min_x, min_y, min_z, 
                                             max_x, max_y, max_z);
    
    // Mark neurons as actuators (filter out dendrite terminals)
    for (const auto& result : search_results) {
        if (!is_terminal_address(result.item_address) && !is_branch_address(result.item_address)) {
            // This is a neuron address
            uint32_t neuron_index = result.item_address >> 12;
            if (neuron_index < MAX_NEURONS) {
                brain.neurons[neuron_index].is_actuator = true;
            }
        }
    }
}

} // namespace neuronlib