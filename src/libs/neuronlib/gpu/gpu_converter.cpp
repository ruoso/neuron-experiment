#include "gpu_converter.h"
#include "../neuron.h"  // For Brain structure
#include "../spatial_operations.h"  // For address operations
#include <spdlog/spdlog.h>

namespace neuronlib {
namespace gpu {

size_t GpuConverter::convert_to_gpu_batches(
    const std::unordered_map<uint32_t, std::vector<Activation>>& pending_messages,
    uint32_t current_timestamp,
    uint32_t timing_window,
    const Brain& brain,
    std::vector<GpuActivationBatch>& output_batches
) {
    output_batches.clear();
    
    if (pending_messages.empty()) {
        return 0;
    }
    
    GpuActivationBatch current_batch;
    current_batch.clear();
    current_batch.current_timestamp = current_timestamp;
    current_batch.timing_window = timing_window;
    
    size_t total_targets_processed = 0;
    size_t total_activations_processed = 0;
    
    for (const auto& [target_address, activations] : pending_messages) {
        // Count valid activations for this target
        size_t valid_count = count_valid_activations(activations, current_timestamp, timing_window);
        
        if (valid_count == 0) {
            continue; // Skip targets with no valid activations
        }
        
        // Check if this target fits in current batch
        if (!current_batch.can_add_target(valid_count)) {
            // Current batch is full, save it and start a new one
            if (current_batch.target_count > 0) {
                output_batches.push_back(current_batch);
                spdlog::debug("GPU batch {} completed: {} targets, {} activations", 
                             output_batches.size(), current_batch.target_count, current_batch.total_activation_count);
            }
            
            current_batch.clear();
            current_batch.current_timestamp = current_timestamp;
            current_batch.timing_window = timing_window;
        }
        
        // Add this target to current batch
        uint32_t target_idx = current_batch.target_count;
        current_batch.target_addresses[target_idx] = target_address;
        current_batch.activation_counts[target_idx] = valid_count;
        current_batch.activation_offsets[target_idx] = current_batch.total_activation_count;
        
        // Look up last activation time for this target
        uint32_t last_activation_index = target_address >> ACTIVATION_TIME_SHIFT;
        current_batch.target_last_activations[target_idx] = brain.last_activations[last_activation_index];
        
        // Look up threshold for neuron targets
        if (is_neuron_address(target_address)) {
            uint32_t neuron_index = target_address >> DENDRITE_ADDRESS_BITS;
            if (neuron_index < MAX_NEURONS) {
                current_batch.target_thresholds[target_idx] = brain.neurons[neuron_index].threshold;
            } else {
                current_batch.target_thresholds[target_idx] = 1.0f; // Default threshold
            }
        } else {
            current_batch.target_thresholds[target_idx] = 0.0f; // Not applicable for non-neuron targets
        }
        
        // Add valid activations to flat arrays
        size_t activation_idx = current_batch.total_activation_count;
        for (const auto& activation : activations) {
            // Check if activation is within timing window
            if (current_timestamp >= activation.timestamp && 
                current_timestamp <= activation.timestamp + timing_window) {
                
                current_batch.activation_values[activation_idx] = activation.value;
                current_batch.activation_timestamps[activation_idx] = activation.timestamp;
                current_batch.source_addresses[activation_idx] = activation.source_address;
                current_batch.activation_weights[activation_idx] = brain.weights[activation.source_address];
                activation_idx++;
            }
        }
        
        current_batch.target_count++;
        current_batch.total_activation_count = activation_idx;
        total_targets_processed++;
        total_activations_processed += valid_count;
    }
    
    // Add final batch if it has data
    if (current_batch.target_count > 0) {
        output_batches.push_back(current_batch);
        spdlog::debug("GPU batch {} completed: {} targets, {} activations", 
                     output_batches.size(), current_batch.target_count, current_batch.total_activation_count);
    }
    
    spdlog::info("Converted CPU data to {} GPU batches: {} targets, {} activations total", 
                output_batches.size(), total_targets_processed, total_activations_processed);
    
    return output_batches.size();
}


size_t GpuConverter::estimate_batch_count(
    const std::unordered_map<uint32_t, std::vector<Activation>>& pending_messages,
    uint32_t current_timestamp,
    uint32_t timing_window
) {
    size_t total_valid_targets = 0;
    size_t total_valid_activations = 0;
    
    for (const auto& [target_address, activations] : pending_messages) {
        size_t valid_count = count_valid_activations(activations, current_timestamp, timing_window);
        if (valid_count > 0) {
            total_valid_targets++;
            total_valid_activations += valid_count;
        }
    }
    
    // Estimate based on both target and activation limits
    size_t batches_by_targets = (total_valid_targets + MAX_GPU_TARGETS - 1) / MAX_GPU_TARGETS;
    size_t batches_by_activations = (total_valid_activations + MAX_GPU_ACTIVATIONS - 1) / MAX_GPU_ACTIVATIONS;
    
    return std::max(batches_by_targets, batches_by_activations);
}

size_t GpuConverter::count_valid_activations(
    const std::vector<Activation>& activations,
    uint32_t current_timestamp,
    uint32_t timing_window
) {
    size_t count = 0;
    for (const auto& activation : activations) {
        if (current_timestamp >= activation.timestamp && 
            current_timestamp <= activation.timestamp + timing_window) {
            count++;
        }
    }
    return count;
}

} // namespace gpu
} // namespace neuronlib