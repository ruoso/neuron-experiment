#include "actuator.h"
#include "brain.h"
#include "spatial_operations.h"
#include "dendrite.h"
#include <algorithm>
#include <limits>
#include <cmath>

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
            uint32_t neuron_index = result.item_address >> DENDRITE_ADDRESS_BITS;
            if (neuron_index < MAX_NEURONS) {
                brain.neurons[neuron_index].is_actuator = true;
            }
        }
    }
}

void mark_actuator_neurons_with_positions(Brain& brain, 
                                         const std::vector<ActuatorPosition>& actuator_positions) {
    if (!brain.spatial_grid || !brain.spatial_grid->root) {
        return;
    }
    
    // First, reset all neurons to not be actuators
    for (size_t i = 0; i < MAX_NEURONS; ++i) {
        brain.neurons[i].is_actuator = false;
        brain.neurons[i].actuator_tag = 0;  // Reset tag
    }
    
    // Distance threshold for considering neurons as actuators
    const float ACTUATOR_DISTANCE_THRESHOLD = 0.2f;
    
    // For each actuator position, find nearby neurons and mark them
    for (const auto& actuator_pos : actuator_positions) {
        // Search for neurons near this actuator position
        float search_radius = ACTUATOR_DISTANCE_THRESHOLD;
        auto search_results = search_bounding_box(brain.spatial_grid,
                                                 actuator_pos.position.x - search_radius,
                                                 actuator_pos.position.y - search_radius,
                                                 actuator_pos.position.z - search_radius,
                                                 actuator_pos.position.x + search_radius,
                                                 actuator_pos.position.y + search_radius,
                                                 actuator_pos.position.z + search_radius);
        
        // Find the closest neuron to this actuator position
        uint32_t closest_neuron_index = MAX_NEURONS;
        float closest_distance = std::numeric_limits<float>::max();
        
        for (const auto& result : search_results) {
            if (!is_terminal_address(result.item_address) && !is_branch_address(result.item_address)) {
                // This is a neuron address
                uint32_t neuron_index = result.item_address >> DENDRITE_ADDRESS_BITS;
                if (neuron_index < MAX_NEURONS) {
                    // Calculate distance from neuron to actuator position
                    Vec3 neuron_pos = brain.neurons[neuron_index].position;
                    float dx = neuron_pos.x - actuator_pos.position.x;
                    float dy = neuron_pos.y - actuator_pos.position.y;
                    float dz = neuron_pos.z - actuator_pos.position.z;
                    float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
                    
                    if (distance < closest_distance) {
                        closest_distance = distance;
                        closest_neuron_index = neuron_index;
                    }
                }
            }
        }
        
        // Mark the closest neuron as an actuator with the tag
        if (closest_neuron_index < MAX_NEURONS && closest_distance <= ACTUATOR_DISTANCE_THRESHOLD) {
            brain.neurons[closest_neuron_index].is_actuator = true;
            brain.neurons[closest_neuron_index].actuator_tag = actuator_pos.tag;
        }
    }
}

} // namespace neuronlib