#include "activation.h"
#include "actuator.h"
#include "dendrite.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>

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
    
    // Process each target address that has pending messages
    auto it = pending_messages_.begin();
    while (it != pending_messages_.end()) {
        uint32_t target_address = it->first;
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
            continue;
        }
        
        // Sum all activations within timing window
        float total_input = 0.0f;
        for (const auto& activation : activations) {
            if (current_timestamp >= activation.timestamp && 
                current_timestamp <= activation.timestamp + timing_window_) {
                total_input += activation.value * brain.weights[target_address];
            }
        }
        
        // Check if this is a terminal or branch address
        if (is_terminal_address(target_address)) {
            // Terminal: propagate to its branch
            uint32_t branch_address = get_terminal_branch(target_address);
            if (total_input > 0.0f) {
                TargetedActivation output(branch_address, Activation(total_input, current_timestamp));
                
                // Check if this goes to same shard or different shard
                if (ShardedMessageProcessor::get_shard_index(branch_address) == 
                    ShardedMessageProcessor::get_shard_index(target_address)) {
                    local_activations.push_back(output);
                } else {
                    cross_shard_activations.push_back(output);
                }
            }
        } else if (is_branch_address(target_address)) {
            // Branch: check threshold and either propagate up or fire neuron
            uint32_t parent_address = get_parent_branch(target_address);
            
            if (parent_address == target_address) {
                // This is the soma (top-level branch)
                uint32_t neuron_address = get_neuron_address(target_address);
                uint32_t neuron_index = neuron_address >> 12;
                
                if (neuron_index < MAX_NEURONS && total_input >= brain.neurons[neuron_index].threshold) {
                    // Neuron fires - check if it's an actuator
                    if (brain.neurons[neuron_index].is_actuator) {
                        // Generate actuation event
                        ActuationEvent actuation_event(brain.neurons[neuron_index].position, current_timestamp);
                        brain.actuation_queue.push(actuation_event);
                    }
                    
                    // Send activations to its output targets
                    for (size_t i = 0; i < MAX_OUTPUT_TARGETS; ++i) {
                        uint32_t output_target = brain.neurons[neuron_index].output_targets[i];
                        if (output_target != 0) {
                            TargetedActivation output(output_target, Activation(1.0f, current_timestamp));
                            
                            // Check if this goes to same shard or different shard
                            if (ShardedMessageProcessor::get_shard_index(output_target) == 
                                ShardedMessageProcessor::get_shard_index(target_address)) {
                                local_activations.push_back(output);
                            } else {
                                cross_shard_activations.push_back(output);
                            }
                        }
                    }
                    
                    // Update last activation time
                    brain.last_activations[neuron_index / 8] = current_timestamp;
                }
            } else {
                // Intermediate branch - propagate to parent
                if (total_input > 0.0f) {
                    TargetedActivation output(parent_address, Activation(total_input, current_timestamp));
                    
                    // Check if this goes to same shard or different shard
                    if (ShardedMessageProcessor::get_shard_index(parent_address) == 
                        ShardedMessageProcessor::get_shard_index(target_address)) {
                        local_activations.push_back(output);
                    } else {
                        cross_shard_activations.push_back(output);
                    }
                }
            }
        }
        
        ++it;
    }
    
    // Handle local activations directly (no thread synchronization needed)
    add_activations(local_activations);
    
    // Send cross-shard activations via thread-safe queues
    if (!cross_shard_activations.empty()) {
        processor->send_activations_to_shards(cross_shard_activations);
    }
}

ShardedMessageProcessor::ShardedMessageProcessor(uint32_t timing_window) {
    for (auto& shard : shards_) {
        shard = ActivationShard(timing_window);
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
    return target_address % NUM_ACTIVATION_SHARDS;
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