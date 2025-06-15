#include "gpu_interface.h"
#include "../brain.h"
#include "../dendrite.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace neuronlib {
namespace gpu {

// CUDA kernel for processing neural activations
__global__ void process_activations_kernel(
    const GpuActivationBatch* input,
    GpuProcessingResults* output
) {
    int target_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (target_idx >= input->target_count) {
        return;
    }
    
    uint32_t target_address = input->target_addresses[target_idx];
    uint32_t activation_count = input->activation_counts[target_idx];
    uint32_t activation_offset = input->activation_offsets[target_idx];
    uint32_t current_timestamp = input->current_timestamp;
    uint32_t timing_window = input->timing_window;
    
    // Process activations within timing window
    float total_input = 0.0f;
    float total_absolute_weights = 0.0f;
    
    for (uint32_t i = 0; i < activation_count; ++i) {
        uint32_t act_idx = activation_offset + i;
        
        if (act_idx >= MAX_GPU_ACTIVATIONS) break;
        
        uint32_t timestamp = input->activation_timestamps[act_idx];
        
        // Check if activation is within timing window
        if (current_timestamp >= timestamp && 
            current_timestamp <= timestamp + timing_window) {
            
            float value = input->activation_values[act_idx];
            float weight = input->activation_weights[act_idx];
            
            float weighted_input = value * weight;
            total_input += weighted_input;
            total_absolute_weights += fabsf(weight);
        }
    }
    
    // Normalize output to 0-1 range based on total absolute weights
    if (total_absolute_weights > 0.0f) {
        total_input = (total_input + total_absolute_weights) / (2.0f * total_absolute_weights);
        total_input = fmaxf(0.0f, fminf(1.0f, total_input));
    }
    
    // Check if this is a neuron address (simplified check for GPU)
    if (is_neuron_address(target_address)) {
        uint32_t neuron_index = target_address >> DENDRITE_ADDRESS_BITS;
        
        if (neuron_index < MAX_NEURONS) {
            float threshold = input->target_thresholds[target_idx];
            bool neuron_fired = total_input >= threshold;
            
            // Check refractory period
            constexpr uint32_t REFRACTORY_PERIOD = 5;
            uint32_t last_firing_time = input->target_last_activations[target_idx];
            bool in_refractory = (current_timestamp - last_firing_time) < REFRACTORY_PERIOD;
            
            if (neuron_fired && in_refractory) {
                neuron_fired = false;
            }
            
            // Add soma activation for CPU processing (regardless of firing)
            uint32_t soma_idx = atomicAdd(&output->soma_count, 1);
            if (soma_idx < MAX_GPU_TARGETS) {
                output->soma_neuron_indices[soma_idx] = neuron_index;
                output->soma_activation_levels[soma_idx] = total_input;
                output->soma_target_addresses[soma_idx] = target_address;
            }
            
            // Apply Hebbian learning - weight updates
            constexpr float LEARNING_RATE = 0.01f;
            for (uint32_t i = 0; i < activation_count; ++i) {
                uint32_t act_idx = activation_offset + i;
                if (act_idx >= MAX_GPU_ACTIVATIONS) break;
                
                uint32_t timestamp = input->activation_timestamps[act_idx];
                if (current_timestamp >= timestamp && 
                    current_timestamp <= timestamp + timing_window) {
                    
                    uint32_t source_address = input->source_addresses[act_idx];
                    float value = input->activation_values[act_idx];
                    float weight = input->activation_weights[act_idx];
                    float weighted_input = value * weight;
                    
                    float delta = 0.0f;
                    if (neuron_fired && weighted_input > 0.0f) {
                        // Strengthen weights that contributed to firing
                        delta = LEARNING_RATE * value;
                    } else if (!neuron_fired) {
                        // Slightly weaken weights when neuron doesn't fire
                        delta = -LEARNING_RATE * 0.1f;
                    }
                    
                    if (delta != 0.0f) {
                        uint32_t weight_idx = atomicAdd(&output->weight_update_count, 1);
                        if (weight_idx < MAX_GPU_WEIGHT_UPDATES) {
                            output->weight_addresses[weight_idx] = source_address;
                            output->weight_deltas[weight_idx] = delta;
                        }
                    }
                }
            }
        }
    } else if (is_terminal_address(target_address)) {
        // Terminal: propagate to its branch
        uint32_t branch_address = get_terminal_branch(target_address);
        if (total_input > 0.0f) {
            uint32_t new_act_idx = atomicAdd(&output->new_activation_count, 1);
            if (new_act_idx < MAX_GPU_OUTPUTS) {
                output->new_activations_target[new_act_idx] = branch_address;
                output->new_activations_value[new_act_idx] = total_input;
                output->new_activations_source[new_act_idx] = target_address;
                output->new_activations_timestamp[new_act_idx] = current_timestamp;
            }
        }
    } else if (is_branch_address(target_address)) {
        // Branch: check threshold and either propagate up or fire neuron
        uint32_t parent_address = get_parent_branch(target_address);
        
        // Intermediate branch - propagate to parent
        if (total_input > 0.0f) {
            uint32_t new_act_idx = atomicAdd(&output->new_activation_count, 1);
            if (new_act_idx < MAX_GPU_OUTPUTS) {
                output->new_activations_target[new_act_idx] = parent_address;
                output->new_activations_value[new_act_idx] = total_input;
                output->new_activations_source[new_act_idx] = target_address;
                output->new_activations_timestamp[new_act_idx] = current_timestamp;
            }
        }
    }
}

} // namespace gpu
} // namespace neuronlib