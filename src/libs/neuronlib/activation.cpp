#include "activation.h"
#include "brain.h"
#include "actuator.h"
#include "dendrite.h"
#include "spatial_operations.h"
#include "geometry.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <new>
#include <random>
#include <limits>

namespace neuronlib {

void ThreadSafeQueue::push_batch(const std::vector<TargetedActivation>& items) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& item : items) {
        queue_.push(item);
    }
}

std::vector<TargetedActivation> ThreadSafeQueue::pop_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<TargetedActivation> result;
    while (!queue_.empty()) {
        result.push_back(queue_.front());
        queue_.pop();
    }
    return result;
}

bool ThreadSafeQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

ActivationShard::ActivationShard(uint32_t timing_window) : timing_window_(timing_window) {}

void ActivationShard::add_activation(uint32_t target_address, const Activation& activation) {
    pending_messages_[target_address].push_back(activation);
}

void ActivationShard::add_activations(const std::vector<TargetedActivation>& activations) {
    for (const auto& activation : activations) {
        pending_messages_[activation.target_address].push_back(activation.activation);
    }
}

ThreadSafeQueue& ActivationShard::get_input_queue() {
    return input_queue_;
}

void ActivationShard::process_tick(Brain& brain, uint32_t current_timestamp, ShardedMessageProcessor* processor) {
    // First, process incoming messages from other shards in batch
    auto incoming_activations = input_queue_.pop_all();
    add_activations(incoming_activations);
    
    // Collect output activations - separate local vs cross-shard
    std::vector<TargetedActivation> local_activations;
    std::vector<TargetedActivation> cross_shard_activations;
    
    std::unordered_map<uint32_t, std::vector<Activation>> new_pending_messages_;

    // Process each target address that has pending messages
    auto it = pending_messages_.begin();
    while (it != pending_messages_.end()) {
        uint32_t target_address = it->first;
        std::vector<Activation>& activations = it->second;
        
        if (activations.empty()) {
            it++;
            continue;  // No activations to process for this address
        }
        
        // Sum all activations within timing window and track which contributed
        float total_input = 0.0f;
        float total_absolute_weights = 0.0f;
        std::vector<bool> activation_contributed(activations.size(), false);
        for (size_t act_idx = 0; act_idx < activations.size(); ++act_idx) {
            const auto& activation = activations[act_idx];
            if (current_timestamp >= activation.timestamp && 
                current_timestamp <= activation.timestamp + timing_window_) {
                float weight = brain.weights[activation.source_address];
                float weighted_input = activation.value * weight;
                total_input += weighted_input;
                total_absolute_weights += std::abs(weight);
                activation_contributed[act_idx] = (weighted_input > 0.0f);
            }
        }
        
        // Normalize output to 0-1 range based on total absolute weights
        if (total_absolute_weights > 0.0f) {
            total_input = (total_input + total_absolute_weights) / (2.0f * total_absolute_weights);
            total_input = std::max(0.0f, std::min(1.0f, total_input));
        }
        
        // Check if this is a terminal or branch address
        if (is_terminal_address(target_address)) {
            // Terminal: propagate to its branch (terminals don't store activation state)
            uint32_t branch_address = get_terminal_branch(target_address);
            if (total_input > 0.0f) {
                TargetedActivation output(branch_address, Activation(total_input, current_timestamp, target_address));
                
                // Check if this goes to same shard or different shard
                if (ShardedMessageProcessor::get_shard_index(branch_address) == 
                    ShardedMessageProcessor::get_shard_index(target_address)) {
                    local_activations.push_back(output);
                } else {
                    cross_shard_activations.push_back(output);
                }
            }
        } else if (is_neuron_address(target_address)) {
            // Direct neuron activation - check threshold and fire
            uint32_t neuron_index = target_address >> DENDRITE_ADDRESS_BITS;
            
            if (neuron_index < MAX_NEURONS) {
                bool neuron_fired = total_input >= brain.neurons[neuron_index].threshold;
                
                // Check refractory period - suppress firing if neuron fired too recently
                constexpr uint32_t REFRACTORY_PERIOD = 5;  // 5 ticks minimum between firings
                uint32_t last_firing_time = brain.last_activations[target_address >> ACTIVATION_TIME_SHIFT];
                bool in_refractory = (current_timestamp - last_firing_time) < REFRACTORY_PERIOD;
                
                if (neuron_fired && in_refractory) {
                    neuron_fired = false;  // Suppress firing due to refractory period
                }
                
                // Adjust weights based on Hebbian learning
                constexpr float LEARNING_RATE = 0.01f;
                for (size_t act_idx = 0; act_idx < activations.size(); ++act_idx) {
                    const auto& activation = activations[act_idx];
                    if (current_timestamp >= activation.timestamp && 
                        current_timestamp <= activation.timestamp + timing_window_) {
                        
                        if (neuron_fired && activation_contributed[act_idx]) {
                            // Strengthen weights that contributed to firing
                            brain.weights[activation.source_address] += LEARNING_RATE * activation.value;
                            brain.weights[activation.source_address] = std::min(brain.weights[activation.source_address], 2.0f); // Cap at 2.0
                        } else if (!neuron_fired) {
                            // Slightly weaken weights when neuron doesn't fire
                            brain.weights[activation.source_address] -= LEARNING_RATE * 0.1f;
                            brain.weights[activation.source_address] = std::max(brain.weights[activation.source_address], 0.1f); // Floor at 0.1
                        }
                    }
                }
                
                if (neuron_fired) {
                    //SPDLOG_DEBUG("Neuron {} firing with total input {}", neuron_index, total_input);
                    
                    // Add to firing events batch for later callback
                    firing_events_batch_.emplace_back(brain.neurons[neuron_index].position, total_input, current_timestamp);
                    
                    // Neuron fires - check if it's an actuator
                    if (brain.neurons[neuron_index].is_actuator) {
                        // Generate actuation event with the actuator tag
                        ActuationEvent actuation_event(brain.neurons[neuron_index].position, current_timestamp, brain.neurons[neuron_index].actuator_tag);
                        SPDLOG_INFO("Neuron {} firing actuator at ({:.3f}, {:.3f}, {:.3f}) with tag {}", 
                                 neuron_index, 
                                 actuation_event.position.x, 
                                 actuation_event.position.y, 
                                 actuation_event.position.z, 
                                 actuation_event.actuator_tag);
                        brain.actuation_queue.push(actuation_event);
                    }
                    
                    // Try to form new connections (only when firing)
                    for (size_t i = 0; i < MAX_OUTPUT_TARGETS; ++i) {
                        if (brain.neurons[neuron_index].output_targets[i] == 0) {
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
                                        result.distance_to_apex < closest_distance &&
                                        brain.weights[result.item_address] == 0.0f // Only unconnected dendrites
                                    ) {
                                        closest_dendrite = result.item_address;
                                        closest_distance = result.distance_to_apex;
                                    }
                                }
                                
                                if (closest_dendrite != 0) {
                                    //SPDLOG_DEBUG("Neuron {} forming new connection to dendrite {}", 
                                    //          neuron_index, closest_dendrite);
                                    brain.neurons[neuron_index].output_targets[i] = closest_dendrite;
                                    // Set the weight for this new connection to a normal distribution
                                    std::normal_distribution<float> weight_dist(0.5f, 0.167f);
                                    brain.weights[closest_dendrite] = weight_dist(connection_rng);
                                }
                            }
                            break; // Only form one connection per activation
                        }
                    }
                    
                    // Send activations to its output targets
                    for (size_t i = 0; i < MAX_OUTPUT_TARGETS; ++i) {
                        uint32_t output_target = brain.neurons[neuron_index].output_targets[i];
                        if (output_target != 0) {
                            // generate the activation in the branch for the terminal we're connected to
                            TargetedActivation output(get_terminal_branch(output_target), Activation(1.0f, current_timestamp, output_target));
                            
                            // Check if this goes to same shard or different shard
                            if (ShardedMessageProcessor::get_shard_index(output_target) == 
                                ShardedMessageProcessor::get_shard_index(target_address)) {
                                local_activations.push_back(output);
                            } else {
                                cross_shard_activations.push_back(output);
                            }
                        }
                    }
                    
                    // Update last activation time (use the target address for proper indexing)
                    brain.last_activations[target_address >> ACTIVATION_TIME_SHIFT] = current_timestamp;
                } else { 
                    // Neuron did not fire, copy the activations to pending messages
                    new_pending_messages_[target_address] = std::move(activations);
                }
            } else {
                spdlog::error("Neuron index {} out of bounds (max {})", neuron_index, MAX_NEURONS);
            }
        } else if (is_branch_address(target_address)) {
            // Branch: check threshold and either propagate up or fire neuron
            uint32_t parent_address = get_parent_branch(target_address);
            
            // Intermediate branch - propagate to parent
            if (total_input > 0.0f) {
                TargetedActivation output(parent_address, Activation(total_input, current_timestamp, target_address));
                
                // Check if this goes to same shard or different shard
                if (ShardedMessageProcessor::get_shard_index(parent_address) == 
                    ShardedMessageProcessor::get_shard_index(target_address)) {
                    local_activations.push_back(output);
                } else {
                    cross_shard_activations.push_back(output);
                }
                
                // apply hebbian learning to this branch
                constexpr float LEARNING_RATE = 0.01f;
                for (size_t act_idx = 0; act_idx < activations.size(); ++act_idx) {
                    const auto& activation = activations[act_idx];
                    if (current_timestamp >= activation.timestamp && 
                        current_timestamp <= activation.timestamp + timing_window_) {
                        
                        // Strengthen weights that contributed to this branch
                        if (activation_contributed[act_idx]) {
                            brain.weights[activation.source_address] += LEARNING_RATE * activation.value;
                            brain.weights[activation.source_address] = std::min(brain.weights[activation.source_address], 2.0f); // Cap at 2.0
                        } else {
                            // Slightly weaken weights when not contributing
                            brain.weights[activation.source_address] -= LEARNING_RATE * 0.1f;
                            brain.weights[activation.source_address] = std::max(brain.weights[activation.source_address], 0.1f); // Floor at 0.1
                        }
                    }
                }

                // Update last activation time for this branch
                brain.last_activations[target_address >> ACTIVATION_TIME_SHIFT] = current_timestamp;
            }
        } else {
            // Unknown address type - log error and skip
            spdlog::error("Unknown target address type: {}", target_address);
        }
        it++;
    }
    
    // Update pending messages with any new activations that didn't fire neurons
    pending_messages_.clear();
    for (const auto& [addr, acts] : new_pending_messages_) {
        // filter out old activations
        std::vector<Activation> filtered_acts;
        for (const auto& act : acts) {
            if (current_timestamp >= act.timestamp && 
                current_timestamp <= act.timestamp + timing_window_) {
                filtered_acts.push_back(act);
            }
        }
        // Only store if there are still valid activations
        if (filtered_acts.empty()) continue;
        // Store the filtered activations back
        // in the pending messages map
        if (filtered_acts.size() == acts.size()) {
            // No filtering needed, just move the original
            pending_messages_[addr] = std::move(acts);
        } else {
            // Store only the filtered activations
            std::vector<Activation> new_acts;
            new_acts.reserve(filtered_acts.size());
            for (const auto& act : filtered_acts) {
                new_acts.push_back(act);
            }
            pending_messages_[addr] = std::move(new_acts);
        }
    }

    // Handle local activations directly (no thread synchronization needed)
    add_activations(local_activations);
    
    // Send cross-shard activations via thread-safe queues
    if (!cross_shard_activations.empty()) {
        processor->send_activations_to_shards(cross_shard_activations);
    }
    
    // Send batched firing events for visualization
    if (!firing_events_batch_.empty() && processor) {
        processor->trigger_neuron_firing_callback(firing_events_batch_);
        firing_events_batch_.clear();
    }
}

ShardedMessageProcessor::ShardedMessageProcessor(uint32_t timing_window) {
    for (size_t i = 0; i < NUM_ACTIVATION_SHARDS; ++i) {
        new(&shards_[i]) ActivationShard(timing_window);
    }
}

void ShardedMessageProcessor::add_activation(uint32_t target_address, const Activation& activation) {
    std::vector<TargetedActivation> batch = {TargetedActivation(target_address, activation)};
    send_activations_to_shards(batch);
}

void ShardedMessageProcessor::send_activations_to_shards(const std::vector<TargetedActivation>& activations) {
    // Group activations by target shard
    std::unordered_map<uint32_t, std::vector<TargetedActivation>> shard_batches;
    
    for (const auto& activation : activations) {
        uint32_t shard_index = get_shard_index(activation.target_address);
        shard_batches[shard_index].push_back(activation);
    }
    
    // Send batches to each shard
    for (const auto& [shard_index, batch] : shard_batches) {
        shards_[shard_index].get_input_queue().push_batch(batch);
    }
}

ActivationShard& ShardedMessageProcessor::get_shard(uint32_t shard_index) {
    return shards_[shard_index];
}

const ActivationShard& ShardedMessageProcessor::get_shard(uint32_t shard_index) const {
    return shards_[shard_index];
}

uint32_t ShardedMessageProcessor::get_shard_index(uint32_t target_address) {
    // Use FNV-1a hash for better distribution
    uint32_t hash = 2166136261u;  // FNV offset basis
    hash ^= target_address;
    hash *= 16777619u;  // FNV prime
    hash ^= (target_address >> 16);
    hash *= 16777619u;
    
    return hash % NUM_ACTIVATION_SHARDS;
}

void ShardedMessageProcessor::set_neuron_firing_callback(NeuronFiringCallback callback) {
    neuron_firing_callback_ = callback;
}

void ShardedMessageProcessor::trigger_neuron_firing_callback(const std::vector<NeuronFiringEvent>& events) {
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