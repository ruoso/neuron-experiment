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

struct Activation {
    float value;
    uint32_t timestamp;
    uint32_t source_address;  // Address of the source sending this activation
    
    Activation() : value(1.0f), timestamp(0), source_address(0) {}
    Activation(float val, uint32_t time, uint32_t source = 0) : value(val), timestamp(time), source_address(source) {}
};


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

#ifdef NEURONLIB_GPU_ENABLED
// Forward declarations for GPU processing
namespace gpu {
    struct GpuActivationBatch;
    struct GpuProcessingResults;
    class GpuConverter;
    bool initialize_gpu();
    void cleanup_gpu();
    bool process_activations_gpu(const GpuActivationBatch& input, 
                                GpuProcessingResults& output);
}
#endif


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


class MessageProcessor {
public:
    MessageProcessor(uint32_t timing_window = 10);
    
    void add_activation(uint32_t target_address, const Activation& activation);
    void send_activations(const std::vector<TargetedActivation>& activations);
    
    void process_tick(Brain& brain, uint32_t current_timestamp);
    
    void set_neuron_firing_callback(NeuronFiringCallback callback);
    void trigger_neuron_firing_callback(const std::vector<NeuronFiringEvent>& events);
    
private:
    uint32_t timing_window_;
    std::unordered_map<uint32_t, std::vector<Activation>> pending_messages_;
    std::vector<NeuronFiringEvent> firing_events_batch_;
    NeuronFiringCallback neuron_firing_callback_;
};


float get_decayed_activation(uint32_t last_activation_time, uint32_t current_time, float decay_rate = 0.9f);

} // namespace neuronlib

#endif // NEURONLIB_ACTIVATION_H