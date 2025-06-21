#include "neural_simulation.h"
#include <spdlog/spdlog.h>
#include <pthread.h>
#include <sched.h>
#include <cstring>
#include <algorithm>

NeuralSimulation::NeuralSimulation() 
    : message_processor_(10), threads_running_(false), simulation_timestamp_(1) {
    
    // Initialize shard completion flags - start as false so timestamp 1 can be processed
    for (auto& completed : shard_completed_) {
        completed.store(false);
    }
    
    last_timestamp_time_ = std::chrono::steady_clock::now();
}

NeuralSimulation::~NeuralSimulation() {
    stop();
}

void NeuralSimulation::initialize() {
    spdlog::info("Initializing neural simulation...");
    
    initialize_brain();
    
    // Set up firing callback
    message_processor_.set_neuron_firing_callback([this](const std::vector<NeuronFiringEvent>& events) {
        std::lock_guard<std::mutex> lock(firing_mutex_);
        for (const auto& event : events) {
            recent_firings_.push_back(event);
        }
        
        // Call external callback if set
        if (firing_callback_) {
            firing_callback_(events);
        }
    });
    
    spdlog::info("Neural simulation initialized successfully");
}

void NeuralSimulation::initialize(const std::vector<SensorPosition>& sensor_positions, 
                                 const std::vector<ActuatorPosition>& actuator_positions) {
    spdlog::info("Initializing neural simulation with custom layout...");
    
    initialize_brain_with_layout(sensor_positions, actuator_positions);
    
    // Set up firing callback
    message_processor_.set_neuron_firing_callback([this](const std::vector<NeuronFiringEvent>& events) {
        std::lock_guard<std::mutex> lock(firing_mutex_);
        for (const auto& event : events) {
            recent_firings_.push_back(event);
        }
        
        // Call external callback if set
        if (firing_callback_) {
            firing_callback_(events);
        }
    });
    
    spdlog::info("Neural simulation with custom layout initialized successfully");
}

void NeuralSimulation::start() {
    if (threads_running_.load()) {
        spdlog::warn("Neural simulation threads already running");
        return;
    }
    
    initialize_shard_threads();
}

void NeuralSimulation::stop() {
    stop_shard_threads();
}

void NeuralSimulation::initialize_brain() {
    spdlog::info("Initializing neural network...");
    
    // Create a simple 3D flow field with more internal processing space
    FlowField3D flow_field(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.3f, 0.2f, 0.1f);
    spdlog::debug("Flow field created: bounds=({}, {}, {}) to ({}, {}, {})", 
                 flow_field.min_x, flow_field.min_y, flow_field.min_z,
                 flow_field.max_x, flow_field.max_y, flow_field.max_z);
    
    // Create brain with matching sensor grid
    brain_ = populate_neuron_grid(flow_field, 1.0f, 45.0f, 0.1f, 0.5f,
                                 GRID_SIZE, GRID_SIZE, 0.3f, 0.3f, 12345);
    
    spdlog::info("Brain initialized successfully:");
    spdlog::info("  - Addressing: {} neuron bits, {} dendrite bits = {} max neurons", 
                 NEURON_ADDRESS_BITS, DENDRITE_ADDRESS_BITS, MAX_NEURONS);
    spdlog::info("  - Sensor grid: {}x{} = {} sensors", GRID_SIZE, GRID_SIZE, GRID_SIZE * GRID_SIZE);
    spdlog::info("  - Neural network ready for processing");
}

void NeuralSimulation::initialize_brain_with_layout(const std::vector<SensorPosition>& sensor_positions, 
                                                   const std::vector<ActuatorPosition>& actuator_positions) {
    spdlog::info("Initializing neural network with custom layout...");
    
    // Create a simple 3D flow field with more internal processing space
    FlowField3D flow_field(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.3f, 0.2f, 0.1f);
    spdlog::debug("Flow field created: bounds=({}, {}, {}) to ({}, {}, {})", 
                 flow_field.min_x, flow_field.min_y, flow_field.min_z,
                 flow_field.max_x, flow_field.max_y, flow_field.max_z);
    
    // Create brain with custom sensor/actuator layout
    brain_ = populate_neuron_grid_with_layout(flow_field, sensor_positions, actuator_positions,
                                             1.0f, 45.0f, 0.1f, 0.5f, 0.3f, 12345);
    
    spdlog::info("Brain with custom layout initialized successfully:");
    spdlog::info("  - Addressing: {} neuron bits, {} dendrite bits = {} max neurons", 
                 NEURON_ADDRESS_BITS, DENDRITE_ADDRESS_BITS, MAX_NEURONS);
    spdlog::info("  - Sensor positions: {} sensors", sensor_positions.size());
    spdlog::info("  - Actuator positions: {} actuators", actuator_positions.size());
    spdlog::info("  - Neural network ready for processing");
}

bool NeuralSimulation::is_ready_to_advance() {
    for (const auto& completed : shard_completed_) {
        if (!completed.load()) {
            return false;
        }
    }
    return true;
}

void NeuralSimulation::advance_timestamp() {
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_timestamp_time_);
    float seconds_per_timestamp = elapsed.count() / 1000.0f;
    
    // Reset all shard completion flags for next timestamp
    for (auto& completed : shard_completed_) {
        completed.store(false);
    }
    simulation_timestamp_.fetch_add(1);
    
    // Log performance for this timestamp
    spdlog::info("Performance: {:.3f} seconds/timestamp (timestamp {})", 
                seconds_per_timestamp, simulation_timestamp_.load());
    
    last_timestamp_time_ = current_time;
}

uint32_t NeuralSimulation::get_current_timestamp() const {
    return simulation_timestamp_.load();
}

void NeuralSimulation::send_sensor_activations(const std::vector<TargetedActivation>& activations) {
    if (!activations.empty()) {
        spdlog::debug("Sending {} sensor activations to neural network", activations.size());
        message_processor_.send_activations_to_shards(activations);
    }
}

std::vector<ActuationEvent> NeuralSimulation::get_actuator_events() {
    return brain_->actuation_queue.pop_all();
}

void NeuralSimulation::set_firing_callback(std::function<void(const std::vector<NeuronFiringEvent>&)> callback) {
    firing_callback_ = callback;
}

std::vector<NeuronFiringEvent> NeuralSimulation::get_recent_firings() {
    std::lock_guard<std::mutex> lock(firing_mutex_);
    std::vector<NeuronFiringEvent> result = recent_firings_;
    recent_firings_.clear();
    return result;
}

const Brain& NeuralSimulation::get_brain() const {
    return *brain_;
}

void NeuralSimulation::initialize_shard_threads() {
    spdlog::info("Initializing shard processing threads...");
    
    threads_running_.store(true);
    
    // Get main thread's current CPU to avoid using it for shards
    cpu_set_t main_cpuset;
    CPU_ZERO(&main_cpuset);
    int main_core = -1;
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &main_cpuset) == 0) {
        // Find the first CPU the main thread is using
        for (int i = 0; i < CPU_SETSIZE; ++i) {
            if (CPU_ISSET(i, &main_cpuset)) {
                main_core = i;
                break;
            }
        }
    }
    
    // Create one thread per shard with CPU affinity
    for (uint32_t shard_idx = 0; shard_idx < NUM_ACTIVATION_SHARDS; ++shard_idx) {
        shard_threads_.emplace_back([this, shard_idx, main_core]() {
            // Set thread affinity to spread across cores
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            
            // Get number of available cores
            int num_cores = std::thread::hardware_concurrency();
            if (num_cores > 1) {  // Need at least 2 cores to avoid main thread
                // Distribute threads across cores, avoiding main thread's core
                int target_core = shard_idx;
                if (main_core >= 0) {
                    // Skip the main core by mapping shard indices to available cores
                    int available_cores = num_cores - 1;
                    target_core = shard_idx % available_cores;
                    if (target_core >= main_core) {
                        target_core++;  // Skip over main core
                    }
                } else {
                    target_core = (shard_idx + 1) % num_cores;  // Fallback: skip core 0
                }
                
                CPU_SET(target_core, &cpuset);
                
                int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                if (result == 0) {
                    spdlog::debug("Shard {} thread bound to CPU core {} (avoiding main core {})", 
                                 shard_idx, target_core, main_core);
                } else {
                    spdlog::warn("Failed to set CPU affinity for shard {} thread: {}", shard_idx, strerror(result));
                }
            }
            
            shard_worker_loop(shard_idx);
        });
    }
    
    spdlog::info("Started {} shard processing threads (avoiding main thread core {})", 
                 NUM_ACTIVATION_SHARDS, main_core);
}

void NeuralSimulation::stop_shard_threads() {
    if (!threads_running_.load()) {
        return;
    }
    
    spdlog::info("Stopping shard processing threads...");
    threads_running_.store(false);
    
    for (auto& thread : shard_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    shard_threads_.clear();
    spdlog::info("All shard threads stopped");
}

void NeuralSimulation::shard_worker_loop(uint32_t shard_idx) {
    spdlog::debug("Shard {} worker thread started", shard_idx);
    
    auto& shard = message_processor_.get_shard(shard_idx);
    uint32_t last_processed_timestamp = 0;
    
    while (threads_running_.load()) {
        uint32_t current_timestamp = simulation_timestamp_.load();
        
        // Only process if we haven't processed this timestamp yet
        if (current_timestamp > last_processed_timestamp) {
            try {
                // Process one tick for this shard
                shard.process_tick(*brain_, current_timestamp, &message_processor_);
                
                // Mark this shard as completed for this timestamp
                shard_completed_[shard_idx].store(true);
                last_processed_timestamp = current_timestamp;
            } catch (const std::exception& e) {
                spdlog::error("Shard {} processing error: {}", shard_idx, e.what());
            }
        } else {
            // Small sleep to prevent excessive CPU usage when waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    spdlog::debug("Shard {} worker thread stopped", shard_idx);
}