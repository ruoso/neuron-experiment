#ifndef NEURAL_SIMULATION_H
#define NEURAL_SIMULATION_H

#include "constants.h"
#include "spatial_operations.h"
#include "flow_field.h"
#include "sensor.h"
#include "actuator.h"
#include "activation.h"
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <array>
#include <functional>

using namespace neuronlib;

class NeuralSimulation {
private:
    // Neural network
    BrainPtr brain_;
    ShardedMessageProcessor message_processor_;
    
    // Threading
    std::vector<std::thread> shard_threads_;
    std::atomic<bool> threads_running_;
    std::atomic<uint32_t> simulation_timestamp_;
    
    // Synchronization for shard processing
    std::array<std::atomic<bool>, NUM_ACTIVATION_SHARDS> shard_completed_;
    
    // Performance monitoring
    std::chrono::steady_clock::time_point last_timestamp_time_;
    
    // Visualization callback
    std::vector<NeuronFiringEvent> recent_firings_;  // Thread-safe accumulator
    std::mutex firing_mutex_;
    
    // Firing callback function
    std::function<void(const std::vector<NeuronFiringEvent>&)> firing_callback_;

public:
    NeuralSimulation();
    ~NeuralSimulation();
    
    void initialize();
    void initialize(const std::vector<SensorPosition>& sensor_positions, 
                   const std::vector<ActuatorPosition>& actuator_positions);
    void start();
    void stop();
    
    // Simulation control
    bool is_ready_to_advance();
    void advance_timestamp();
    uint32_t get_current_timestamp() const;
    
    // Sensor interaction
    void send_sensor_activations(const std::vector<TargetedActivation>& activations);
    
    // Actuator interaction
    std::vector<ActuationEvent> get_actuator_events();
    
    // Firing callback
    void set_firing_callback(std::function<void(const std::vector<NeuronFiringEvent>&)> callback);
    std::vector<NeuronFiringEvent> get_recent_firings();
    
    // Brain access
    const Brain& get_brain() const;

private:
    void initialize_brain();
    void initialize_brain_with_layout(const std::vector<SensorPosition>& sensor_positions, 
                                     const std::vector<ActuatorPosition>& actuator_positions);
    void initialize_shard_threads();
    void stop_shard_threads();
    void shard_worker_loop(uint32_t shard_idx);
};

#endif