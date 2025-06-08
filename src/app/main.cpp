#include <SDL2/SDL.h>
#include "spatial_operations.h"
#include "flow_field.h"
#include "sensor.h"
#include "actuator.h"
#include "activation.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

using namespace neuronlib;

constexpr int WINDOW_WIDTH = 800;
constexpr int WINDOW_HEIGHT = 600;
constexpr int GRID_SIZE = 32;
constexpr int CELL_SIZE = std::min(WINDOW_WIDTH, WINDOW_HEIGHT) / GRID_SIZE;
constexpr int GRID_OFFSET_X = (WINDOW_WIDTH - GRID_SIZE * CELL_SIZE) / 2;
constexpr int GRID_OFFSET_Y = (WINDOW_HEIGHT - GRID_SIZE * CELL_SIZE) / 2;

enum class CellSource {
    NONE = 0,
    USER = 1,
    ACTUATOR = 2
};

struct GridCell {
    float user_intensity;      // 0.0 to 1.0
    float actuator_intensity;  // 0.0 to 1.0
    CellSource last_source;
    
    GridCell() : user_intensity(0.0f), actuator_intensity(0.0f), last_source(CellSource::NONE) {}
    
    float get_total_intensity() const {
        return std::max(user_intensity, actuator_intensity);
    }
    
    uint8_t get_gray_value() const {
        return static_cast<uint8_t>(get_total_intensity() * 255);
    }
};

class NeuronExperimentApp {
private:
    SDL_Window* window_;
    SDL_Renderer* renderer_;
    bool running_;
    
    // Grid state
    std::vector<std::vector<GridCell>> grid_;
    
    // Neural network
    BrainPtr brain_;
    ShardedMessageProcessor message_processor_;
    
    // Threading
    std::vector<std::thread> shard_threads_;
    std::atomic<bool> threads_running_;
    std::atomic<uint32_t> simulation_timestamp_;
    
    // Timing
    std::chrono::steady_clock::time_point last_update_;
    
public:
    NeuronExperimentApp() : window_(nullptr), renderer_(nullptr), running_(false), 
                           grid_(GRID_SIZE, std::vector<GridCell>(GRID_SIZE)),
                           message_processor_(10),
                           threads_running_(false),
                           simulation_timestamp_(0) {
        
        // Initialize logging
        initialize_logging();
        
        // Initialize neural network
        initialize_brain();
        
        // Initialize shard threads
        initialize_shard_threads();
        
        last_update_ = std::chrono::steady_clock::now();
    }
    
    ~NeuronExperimentApp() {
        stop_shard_threads();
        cleanup();
    }
    
    void initialize_logging() {
        try {
            // Create console sink with colors
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_level(spdlog::level::info);
            console_sink->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
            
            // Create file sink
            auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("neuron_experiment.log", true);
            file_sink->set_level(spdlog::level::debug);
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v");
            
            // Create logger with both sinks
            auto logger = std::make_shared<spdlog::logger>("neuron_app", 
                                                          spdlog::sinks_init_list{console_sink, file_sink});
            logger->set_level(spdlog::level::debug);
            
            // Set as default logger
            spdlog::set_default_logger(logger);
            
            spdlog::info("Logging system initialized");
            spdlog::debug("Debug logging enabled to file: neuron_experiment.log");
        } catch (const spdlog::spdlog_ex& ex) {
            std::cerr << "Log initialization failed: " << ex.what() << std::endl;
        }
    }
    
    bool initialize() {
        spdlog::info("Initializing Neuron Experiment Application...");
        
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            spdlog::error("SDL initialization failed: {}", SDL_GetError());
            return false;
        }
        spdlog::debug("SDL initialized successfully");
        
        window_ = SDL_CreateWindow("Neuron Experiment",
                                 SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                 WINDOW_WIDTH, WINDOW_HEIGHT,
                                 SDL_WINDOW_SHOWN);
        
        if (!window_) {
            spdlog::error("Window creation failed: {}", SDL_GetError());
            return false;
        }
        spdlog::debug("Window created: {}x{}", WINDOW_WIDTH, WINDOW_HEIGHT);
        
        renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED);
        if (!renderer_) {
            spdlog::error("Renderer creation failed: {}", SDL_GetError());
            return false;
        }
        spdlog::debug("Renderer created successfully");
        
        spdlog::info("Application initialization complete");
        return true;
    }
    
    void initialize_brain() {
        spdlog::info("Initializing neural network...");
        
        // Create a simple 3D flow field
        FlowField3D flow_field(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.4f, 0.4f, 0.1f);
        spdlog::debug("Flow field created: bounds=({}, {}, {}) to ({}, {}, {})", 
                     flow_field.min_x, flow_field.min_y, flow_field.min_z,
                     flow_field.max_x, flow_field.max_y, flow_field.max_z);
        
        // Create brain with matching sensor grid
        brain_ = populate_neuron_grid(flow_field, 1.0f, 45.0f, 0.1f, 0.5f,
                                     GRID_SIZE, GRID_SIZE, 0.3f, 0.1f, 12345);
        
        spdlog::info("Brain initialized successfully:");
        spdlog::info("  - Addressing: {} neuron bits, {} dendrite bits = {} max neurons", 
                     NEURON_ADDRESS_BITS, DENDRITE_ADDRESS_BITS, MAX_NEURONS);
        spdlog::info("  - Sensor grid: {}x{} = {} sensors", GRID_SIZE, GRID_SIZE, GRID_SIZE * GRID_SIZE);
        spdlog::info("  - Neural network ready for processing");
    }
    
    void initialize_shard_threads() {
        spdlog::info("Initializing shard processing threads...");
        
        threads_running_.store(true);
        
        // Create one thread per shard
        for (uint32_t shard_idx = 0; shard_idx < NUM_ACTIVATION_SHARDS; ++shard_idx) {
            shard_threads_.emplace_back([this, shard_idx]() {
                shard_worker_loop(shard_idx);
            });
        }
        
        spdlog::info("Started {} shard processing threads", NUM_ACTIVATION_SHARDS);
    }
    
    void stop_shard_threads() {
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
    
    void shard_worker_loop(uint32_t shard_idx) {
        spdlog::debug("Shard {} worker thread started", shard_idx);
        
        auto& shard = message_processor_.get_shard(shard_idx);
        
        while (threads_running_.load()) {
            uint32_t current_timestamp = simulation_timestamp_.load();
            
            // Process one tick for this shard
            shard.process_tick(*brain_, current_timestamp, &message_processor_);
            
            // Small sleep to prevent excessive CPU usage
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        
        spdlog::debug("Shard {} worker thread stopped", shard_idx);
    }
    
    void run() {
        if (!initialize()) {
            return;
        }
        
        running_ = true;
        
        while (running_) {
            handle_events();
            update();
            render();
            
            SDL_Delay(16); // ~60 FPS
        }
        
        stop_shard_threads();
    }
    
    void handle_events() {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    running_ = false;
                    break;
                
                case SDL_MOUSEBUTTONDOWN:
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        handle_mouse_click(event.button.x, event.button.y);
                    }
                    break;
                
                case SDL_MOUSEMOTION:
                    if (event.motion.state & SDL_BUTTON_LMASK) {
                        handle_mouse_click(event.motion.x, event.motion.y);
                    }
                    break;
            }
        }
    }
    
    void handle_mouse_click(int x, int y) {
        // Convert screen coordinates to grid coordinates
        int grid_x = (x - GRID_OFFSET_X) / CELL_SIZE;
        int grid_y = (y - GRID_OFFSET_Y) / CELL_SIZE;
        
        if (grid_x >= 0 && grid_x < GRID_SIZE && grid_y >= 0 && grid_y < GRID_SIZE) {
            grid_[grid_y][grid_x].user_intensity = 1.0f;
            grid_[grid_y][grid_x].last_source = CellSource::USER;
            spdlog::debug("User activated cell ({}, {}) at screen pos ({}, {})", grid_x, grid_y, x, y);
        }
    }
    
    void update() {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_update_);
        
        if (elapsed.count() >= 100) {
            simulation_step();
            last_update_ = current_time;
            simulation_timestamp_.fetch_add(1);
        }
    }
    
    void simulation_step() {
        uint32_t current_timestamp = simulation_timestamp_.load();
        spdlog::debug("=== Simulation Step {} ===", current_timestamp);
        
        // 1. Fade all cells by 1/4
        fade_grid();
        
        // 2. Generate sensor activations for non-black cells
        generate_sensor_activations();
        
        // 3. Process neural network for one step
        process_neural_network();
        
        // 4. Handle actuator outputs
        process_actuator_outputs();
        
        spdlog::debug("Simulation step {} complete", current_timestamp);
    }
    
    void fade_grid() {
        for (auto& row : grid_) {
            for (auto& cell : row) {
                cell.user_intensity *= 0.75f;
                cell.actuator_intensity *= 0.75f;
                
                if (cell.get_total_intensity() < 0.01f) {
                    cell.user_intensity = 0.0f;
                    cell.actuator_intensity = 0.0f;
                    cell.last_source = CellSource::NONE;
                }
            }
        }
    }
    
    void generate_sensor_activations() {
        std::vector<SensorActivation> activations;
        int user_activations = 0;
        int actuator_activations = 0;
        
        for (int y = 0; y < GRID_SIZE; ++y) {
            for (int x = 0; x < GRID_SIZE; ++x) {
                const auto& cell = grid_[y][x];
                
                if (cell.get_total_intensity() > 0.01f) {
                    uint32_t sensor_index = y * GRID_SIZE + x;
                    
                    // Determine which modes to activate based on intensity and source
                    uint8_t mode_bitmap = 0;
                    
                    if (cell.user_intensity > 0.01f) {
                        // User input: use modes 0-3 based on intensity
                        float intensity = cell.user_intensity;
                        if (intensity > 0.75f) mode_bitmap |= (1 << 0);  // Mode 0: Very bright
                        else if (intensity > 0.5f) mode_bitmap |= (1 << 1);   // Mode 1: Bright
                        else if (intensity > 0.25f) mode_bitmap |= (1 << 2);  // Mode 2: Medium
                        else mode_bitmap |= (1 << 3);                         // Mode 3: Dim
                        user_activations++;
                    }
                    
                    if (cell.actuator_intensity > 0.01f) {
                        // Actuator feedback: use all 4 modes to indicate self-generated
                        mode_bitmap = 0xF; // All 4 modes active
                        actuator_activations++;
                    }
                    
                    if (mode_bitmap != 0) {
                        activations.emplace_back(sensor_index, mode_bitmap, cell.get_total_intensity());
                    }
                }
            }
        }
        
        if (!activations.empty()) {
            spdlog::debug("Generated {} sensor activations: {} user, {} actuator", 
                         activations.size(), user_activations, actuator_activations);
        }
        
        // Process sensor activations and send to neural network
        uint32_t current_timestamp = simulation_timestamp_.load();
        auto targeted_activations = process_sensor_activations(brain_->sensor_grid, activations, current_timestamp);
        if (!targeted_activations.empty()) {
            spdlog::debug("Converted to {} targeted neural activations", targeted_activations.size());
            message_processor_.send_activations_to_shards(targeted_activations);
        }
    }
    
    void process_neural_network() {
        // Neural network processing is now handled by shard threads
        // Just update the simulation timestamp to coordinate the threads
        spdlog::debug("Updating simulation timestamp for {} shard threads", NUM_ACTIVATION_SHARDS);
    }
    
    void process_actuator_outputs() {
        // Get all actuation events
        auto actuation_events = brain_->actuation_queue.pop_all();
        
        if (!actuation_events.empty()) {
            spdlog::info("Processing {} actuator outputs", actuation_events.size());
        }
        
        for (const auto& event : actuation_events) {
            // Convert world position to grid coordinates
            // Assuming the sensor grid spans the same area as the flow field
            float norm_x = (event.position.x - (-1.0f)) / (1.0f - (-1.0f));  // Normalize to 0-1
            float norm_y = (event.position.y - (-1.0f)) / (1.0f - (-1.0f));  // Normalize to 0-1
            
            int grid_x = static_cast<int>(norm_x * GRID_SIZE);
            int grid_y = static_cast<int>(norm_y * GRID_SIZE);
            
            // Clamp to grid bounds
            grid_x = std::max(0, std::min(GRID_SIZE - 1, grid_x));
            grid_y = std::max(0, std::min(GRID_SIZE - 1, grid_y));
            
            spdlog::debug("Actuator event: world_pos=({:.2f}, {:.2f}, {:.2f}) -> grid=({}, {})", 
                         event.position.x, event.position.y, event.position.z, grid_x, grid_y);
            
            // Set actuator intensity
            grid_[grid_y][grid_x].actuator_intensity = 1.0f;
            grid_[grid_y][grid_x].last_source = CellSource::ACTUATOR;
        }
    }
    
    void render() {
        // Clear screen to black
        SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
        SDL_RenderClear(renderer_);
        
        // Draw grid
        for (int y = 0; y < GRID_SIZE; ++y) {
            for (int x = 0; x < GRID_SIZE; ++x) {
                const auto& cell = grid_[y][x];
                
                SDL_Rect rect;
                rect.x = GRID_OFFSET_X + x * CELL_SIZE;
                rect.y = GRID_OFFSET_Y + y * CELL_SIZE;
                rect.w = CELL_SIZE - 1;  // Leave 1 pixel border
                rect.h = CELL_SIZE - 1;
                
                uint8_t gray = cell.get_gray_value();
                
                // Color coding: user input = white/gray, actuator = colored
                if (cell.last_source == CellSource::ACTUATOR && cell.actuator_intensity > 0.01f) {
                    // Actuator output in blue-ish color
                    SDL_SetRenderDrawColor(renderer_, 0, gray/2, gray, 255);
                } else {
                    // User input or faded in grayscale
                    SDL_SetRenderDrawColor(renderer_, gray, gray, gray, 255);
                }
                
                SDL_RenderFillRect(renderer_, &rect);
            }
        }
        
        SDL_RenderPresent(renderer_);
    }
    
    void cleanup() {
        if (renderer_) {
            SDL_DestroyRenderer(renderer_);
            renderer_ = nullptr;
        }
        
        if (window_) {
            SDL_DestroyWindow(window_);
            window_ = nullptr;
        }
        
        SDL_Quit();
    }
};

int main(int argc, char* argv[]) {
    NeuronExperimentApp app;
    app.run();
    return 0;
}