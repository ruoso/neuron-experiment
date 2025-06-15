#include "activation.h"
#include "brain.h"
#include "actuator.h"
#include "dendrite.h"
#include "spatial_operations.h"
#include "geometry.h"
#include "gpu/gpu_interface.h"
#include "gpu/gpu_converter.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <new>
#include <random>
#include <limits>

namespace neuronlib {



MessageProcessor::MessageProcessor(uint32_t timing_window) : timing_window_(timing_window) {
    // Initialize GPU processing
    if (!gpu::initialize_gpu()) {
        spdlog::error("Failed to initialize GPU - this will cause runtime errors");
    } else {
        spdlog::info("GPU processing initialized successfully");
    }
}

void MessageProcessor::add_activation(uint32_t target_address, const Activation& activation) {
    pending_messages_[target_address].push_back(activation);
}

void MessageProcessor::send_activations(const std::vector<TargetedActivation>& activations) {
    for (const auto& activation : activations) {
        pending_messages_[activation.target_address].push_back(activation.activation);
    }
}

void MessageProcessor::process_tick(Brain& brain, uint32_t current_timestamp) {
    if (pending_messages_.empty()) {
        return;
    }
    
    // Convert pending messages to GPU batches
    std::vector<gpu::GpuActivationBatch> gpu_batches;
    size_t batch_count = gpu::GpuConverter::convert_to_gpu_batches(
        pending_messages_, current_timestamp, timing_window_, brain, gpu_batches
    );
    
    if (batch_count == 0) {
        spdlog::debug("No valid activations to process");
        return;
    }
    
    // Process each batch on GPU
    std::vector<gpu::GpuProcessingResults> gpu_results(batch_count);
    for (size_t i = 0; i < batch_count; ++i) {
        bool success = gpu::process_activations_gpu(
            gpu_batches[i], gpu_results[i]
        );
        
        if (!success) {
            spdlog::error("GPU processing failed for batch {}", i);
            continue;
        }
    }
    
    // Clear processed pending messages - GPU has processed all valid activations
    // Remove old activations outside timing window and empty entries
    auto it = pending_messages_.begin();
    while (it != pending_messages_.end()) {
        std::vector<Activation>& activations = it->second;
        
        // Remove old activations outside timing window
        activations.erase(
            std::remove_if(activations.begin(), activations.end(),
                [current_timestamp, this](const Activation& act) {
                    return current_timestamp > act.timestamp + timing_window_;
                }),
            activations.end()
        );
        
        if (activations.empty()) {
            it = pending_messages_.erase(it);
        } else {
            ++it;
        }
    }
    
    // Process GPU results
    size_t total_firings = 0;
    size_t total_new_activations = 0;
    size_t total_actuations = 0;
    size_t total_weight_updates = 0;
    
    for (const auto& result : gpu_results) {
        // Process weight updates
        for (uint32_t i = 0; i < result.weight_update_count; ++i) {
            uint32_t address = result.weight_addresses[i];
            float delta = result.weight_deltas[i];
            
            // Apply weight update with bounds checking
            brain.weights[address] += delta;
            brain.weights[address] = std::max(0.1f, std::min(2.0f, brain.weights[address]));
            total_weight_updates++;
        }
        
        // Process neuron firings and generate their outputs
        for (uint32_t i = 0; i < result.firing_count; ++i) {
            uint32_t neuron_index = result.fired_neurons[i];
            float intensity = result.firing_intensities[i];
            
            if (neuron_index >= MAX_NEURONS) {
                spdlog::warn("Invalid neuron index from GPU: {}", neuron_index);
                continue;
            }
            
            const Neuron& neuron = brain.neurons[neuron_index];
            
            // Create firing event for visualization
            firing_events_batch_.emplace_back(neuron.position, intensity, current_timestamp);
            
            // Generate output activations to connected neurons
            for (size_t target_idx = 0; target_idx < MAX_OUTPUT_TARGETS; ++target_idx) {
                uint32_t output_target = neuron.output_targets[target_idx];
                if (output_target != 0) {
                    // Create new activation with neuron index as source
                    uint32_t source_address = neuron_index << DENDRITE_ADDRESS_BITS;
                    pending_messages_[output_target].emplace_back(1.0f, current_timestamp, source_address);
                    total_new_activations++;
                } else {
                    // Empty slot - small chance to form a new connection (80% chance)
                    static std::mt19937 connection_rng(current_timestamp);
                    std::uniform_real_distribution<float> connection_chance(0.0f, 1.0f);
            
                    if (connection_chance(connection_rng) < 0.8f) {
                        // Try to find a dendrite in the output cone
                        Vec3 neuron_pos = brain.neurons[neuron_index].position;
                        Vec3 output_dir = brain.neurons[neuron_index].output_direction;
                        
                        // Create frustum to search for dendrites in output direction
                        Vec3 cone_end = {
                            neuron_pos.x + output_dir.x * 1.0f,  // 1.0 unit forward
                            neuron_pos.y + output_dir.y * 1.0f,
                            neuron_pos.z + output_dir.z * 1.0f
                        };
                        Frustum3D search_cone(neuron_pos, cone_end, 0.0f, 0.3f);  // Small apex, wider base
                        
                        // Search for dendrites in the cone
                        auto search_results = search_frustum(brain.spatial_grid, search_cone);
                        
                        // Find the closest dendrite terminal
                        uint32_t closest_dendrite = 0;
                        float closest_distance = std::numeric_limits<float>::max();
                        
                        for (const auto& result : search_results) {
                            if (is_terminal_address(result.item_address) && 
                                result.distance_to_apex < closest_distance) {
                                closest_dendrite = result.item_address;
                                closest_distance = result.distance_to_apex;
                            }
                        }
                        
                        if (closest_dendrite != 0) {
                            brain.neurons[neuron_index].output_targets[i] = closest_dendrite;
                        }
                    }
                    break; // Only attempt to form one connection per activation
                }
            }
            
            // Update last activation time
            uint32_t address_for_timing = neuron_index << DENDRITE_ADDRESS_BITS;
            brain.last_activations[address_for_timing >> ACTIVATION_TIME_SHIFT] = current_timestamp;
            
            total_firings++;
        }
        
        // Process actuator firings
        for (uint32_t i = 0; i < result.actuator_count; ++i) {
            uint32_t neuron_index = result.actuator_neurons[i];
            
            if (neuron_index < MAX_NEURONS) {
                const Neuron& neuron = brain.neurons[neuron_index];
                ActuationEvent actuation_event(neuron.position, current_timestamp);
                brain.actuation_queue.push(actuation_event);
                total_actuations++;
            }
        }
        
        // Process additional new activations from GPU (terminals, branches, etc.)
        for (uint32_t i = 0; i < result.new_activation_count; ++i) {
            uint32_t target = result.new_activations_target[i];
            float value = result.new_activations_value[i];
            uint32_t source = result.new_activations_source[i];
            uint32_t timestamp = result.new_activations_timestamp[i];
            
            pending_messages_[target].emplace_back(value, timestamp, source);
            total_new_activations++;
        }
    }
    
    spdlog::debug("GPU processing completed: {} firings, {} new activations, {} actuations, {} weight updates",
                 total_firings, total_new_activations, total_actuations, total_weight_updates);
    
    // Send firing events to callback
    if (!firing_events_batch_.empty()) {
        trigger_neuron_firing_callback(firing_events_batch_);
        firing_events_batch_.clear();
    }        
}

void MessageProcessor::set_neuron_firing_callback(NeuronFiringCallback callback) {
    neuron_firing_callback_ = callback;
}

void MessageProcessor::trigger_neuron_firing_callback(const std::vector<NeuronFiringEvent>& events) {
    if (neuron_firing_callback_ && !events.empty()) {
        neuron_firing_callback_(events);
    }
}


float get_decayed_activation(uint32_t last_activation_time, uint32_t current_time, float decay_rate) {
    if (current_time <= last_activation_time) {
        return 1.0f;
    }
    
    uint32_t time_diff = current_time - last_activation_time;
    return std::pow(decay_rate, static_cast<float>(time_diff));
}

void ActuationQueue::push(const ActuationEvent& event) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(event);
}

void ActuationQueue::push_batch(const std::vector<ActuationEvent>& events) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& event : events) {
        queue_.push(event);
    }
}

std::vector<ActuationEvent> ActuationQueue::pop_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ActuationEvent> result;
    while (!queue_.empty()) {
        result.push_back(queue_.front());
        queue_.pop();
    }
    return result;
}

bool ActuationQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

} // namespace neuronlib