#ifndef NEURONLIB_ACTIVATION_H
#define NEURONLIB_ACTIVATION_H

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <array>
#include <queue>
#include <mutex>
#include <atomic>
#include <functional>
#include "geometry.h"

namespace neuronlib {

// Forward declarations
struct Brain;
struct ActuationEvent;
class ShardedMessageProcessor;

struct Activation {
    float value;
    uint32_t timestamp;
    
    Activation() : value(1.0f), timestamp(0) {}
    Activation(float val, uint32_t time) : value(val), timestamp(time) {}
};

constexpr size_t NUM_ACTIVATION_SHARDS = 15;

struct TargetedActivation {
    uint32_t target_address;
    Activation activation;
    
    TargetedActivation() : target_address(0) {}
    TargetedActivation(uint32_t addr, const Activation& act) : target_address(addr), activation(act) {}
};

struct NeuronFiringEvent {
    Vec3 position;
    float activation_strength;
    uint32_t timestamp;
    
    NeuronFiringEvent(const Vec3& pos, float strength, uint32_t time) 
        : position(pos), activation_strength(strength), timestamp(time) {}
};

using NeuronFiringCallback = std::function<void(const std::vector<NeuronFiringEvent>&)>;

class ThreadSafeQueue {
public:
    void push_batch(const std::vector<TargetedActivation>& items);
    std::vector<TargetedActivation> pop_all();
    bool empty() const;
    
private:
    mutable std::mutex mutex_;
    std::queue<TargetedActivation> queue_;
};

class ActuationQueue {
public:
    void push(const struct ActuationEvent& event);
    void push_batch(const std::vector<struct ActuationEvent>& events);
    std::vector<struct ActuationEvent> pop_all();
    bool empty() const;
    
private:
    mutable std::mutex mutex_;
    std::queue<struct ActuationEvent> queue_;
};

class ActivationShard {
public:
    ActivationShard(uint32_t timing_window = 10);
    
    void add_activation(uint32_t target_address, const Activation& activation);
    void add_activations(const std::vector<TargetedActivation>& activations);
    
    void process_tick(Brain& brain, uint32_t current_timestamp, ShardedMessageProcessor* processor);
    
    ThreadSafeQueue& get_input_queue();
    
private:
    uint32_t timing_window_;
    std::unordered_map<uint32_t, std::vector<Activation>> pending_messages_;
    ThreadSafeQueue input_queue_;
    std::vector<NeuronFiringEvent> firing_events_batch_;
};

class ShardedMessageProcessor {
public:
    ShardedMessageProcessor(uint32_t timing_window = 10);
    
    void add_activation(uint32_t target_address, const Activation& activation);
    
    void send_activations_to_shards(const std::vector<TargetedActivation>& activations);
    
    ActivationShard& get_shard(uint32_t shard_index);
    const ActivationShard& get_shard(uint32_t shard_index) const;
    
    static uint32_t get_shard_index(uint32_t target_address);
    
    void set_neuron_firing_callback(NeuronFiringCallback callback);
    void trigger_neuron_firing_callback(const std::vector<NeuronFiringEvent>& events);
    
private:
    std::array<ActivationShard, NUM_ACTIVATION_SHARDS> shards_;
    NeuronFiringCallback neuron_firing_callback_;
};

float get_decayed_activation(uint32_t last_activation_time, uint32_t current_time, float decay_rate = 0.9f);

} // namespace neuronlib

#endif // NEURONLIB_ACTIVATION_H