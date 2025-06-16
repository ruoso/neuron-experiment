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
#include <pthread.h>
#include <sched.h>
#include <cstring>

using namespace neuronlib;

constexpr int WINDOW_WIDTH = 800;
constexpr int WINDOW_HEIGHT = 600;
constexpr int VIZ_WINDOW_WIDTH = 900;
constexpr int VIZ_WINDOW_HEIGHT = 900;
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
    bool activation_sent;      // Track if activation was already sent
    
    GridCell() : user_intensity(0.0f), actuator_intensity(0.0f), last_source(CellSource::NONE), activation_sent(false) {}
    
    float get_total_intensity() const {
        return std::max(user_intensity, actuator_intensity);
    }
    
    uint8_t get_gray_value() const {
        return static_cast<uint8_t>(get_total_intensity() * 255);
    }
};

struct RippleEffect {
    int center_x, center_y;
    float current_radius;
    float max_radius;
    CellSource source_type;
    uint32_t start_time;
    
    RippleEffect(int x, int y, CellSource source, uint32_t time) 
        : center_x(x), center_y(y), current_radius(0.0f), max_radius(4.0f), 
          source_type(source), start_time(time) {}
};

struct IsometricPoint {
    float x, y;
};

class NeuronExperimentApp {
private:
    SDL_Window* window_;
    SDL_Renderer* renderer_;
    SDL_Window* viz_window_;
    SDL_Renderer* viz_renderer_;
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
    
    // Synchronization for shard processing
    std::array<std::atomic<bool>, NUM_ACTIVATION_SHARDS> shard_completed_;
    
    // Performance monitoring
    std::chrono::steady_clock::time_point last_timestamp_time_;
    
    // Visualization
    std::vector<NeuronFiringEvent> recent_firings_;  // Thread-safe accumulator
    std::mutex firing_mutex_;
    
    // Timing
    std::chrono::steady_clock::time_point last_update_;
    
    // Z-coordinate tracking for chart
    static constexpr int CHART_HISTORY_SIZE = 50;  // 50 simulation timestamps of history
    static constexpr int Z_BINS = 20;              // Number of Z-coordinate bins
    std::vector<std::vector<int>> z_distribution_history_;  // [time_index][z_bin]
    std::vector<int> sensor_activation_history_;           // [time_index] -> count of sensor activations
    std::vector<float> max_z_history_;                     // [time_index] -> max Z coordinate of activations
    int current_history_index_;
    
    // 3D visualization time bins for fading effect
    static constexpr int FIRING_TIME_BINS = 10;     // 10 bins for 1 second at 100ms intervals
    std::vector<std::vector<Vec3>> firing_time_bins_;  // [time_bin][positions]
    int current_firing_bin_;
    
    // Ripple effects
    std::vector<RippleEffect> active_ripples_;
    
    // Scheduled activations
    static constexpr int SCHEDULE_POINTS = 12;           // 12 points around the circle
    static constexpr uint32_t SCHEDULE_INTERVAL = 100;   // 10 seconds = 100 simulation steps
    uint32_t next_scheduled_activation_;
    int current_schedule_point_;
    
public:
    NeuronExperimentApp() : window_(nullptr), renderer_(nullptr), viz_window_(nullptr), viz_renderer_(nullptr),
                           running_(false), grid_(GRID_SIZE, std::vector<GridCell>(GRID_SIZE)),
                           message_processor_(10), threads_running_(false), simulation_timestamp_(1),
                           z_distribution_history_(CHART_HISTORY_SIZE, std::vector<int>(Z_BINS, 0)),
                           sensor_activation_history_(CHART_HISTORY_SIZE, 0),
                           max_z_history_(CHART_HISTORY_SIZE, -2.0f),
                           current_history_index_(0),
                           firing_time_bins_(FIRING_TIME_BINS),
                           current_firing_bin_(0),
                           next_scheduled_activation_(SCHEDULE_INTERVAL),
                           current_schedule_point_(0) {
        
        // Initialize logging
        initialize_logging();
        
        // Initialize neural network
        initialize_brain();
        
        // Set up firing callback
        message_processor_.set_neuron_firing_callback([this](const std::vector<NeuronFiringEvent>& events) {
            std::lock_guard<std::mutex> lock(firing_mutex_);
            for (const auto& event : events) {
                recent_firings_.push_back(event);
            }
        });
        
        // Initialize shard completion flags - start as false so timestamp 1 can be processed
        for (auto& completed : shard_completed_) {
            completed.store(false);
        }
        
        // Initialize shard threads
        initialize_shard_threads();
        
        last_update_ = std::chrono::steady_clock::now();
        last_timestamp_time_ = std::chrono::steady_clock::now();
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
        
        window_ = SDL_CreateWindow("Neuron Experiment - 2D Grid",
                                 SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                 WINDOW_WIDTH, WINDOW_HEIGHT,
                                 SDL_WINDOW_SHOWN);
        
        if (!window_) {
            spdlog::error("Main window creation failed: {}", SDL_GetError());
            return false;
        }
        spdlog::debug("Main window created: {}x{}", WINDOW_WIDTH, WINDOW_HEIGHT);
        
        renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED);
        if (!renderer_) {
            spdlog::error("Main renderer creation failed: {}", SDL_GetError());
            return false;
        }
        spdlog::debug("Main renderer created successfully");
        
        // Create visualization window
        viz_window_ = SDL_CreateWindow("Neuron Experiment - 3D Visualization",
                                     WINDOW_WIDTH + 50, SDL_WINDOWPOS_UNDEFINED,
                                     VIZ_WINDOW_WIDTH, VIZ_WINDOW_HEIGHT,
                                     SDL_WINDOW_SHOWN);
        
        if (!viz_window_) {
            spdlog::error("Visualization window creation failed: {}", SDL_GetError());
            return false;
        }
        spdlog::debug("Visualization window created: {}x{}", VIZ_WINDOW_WIDTH, VIZ_WINDOW_HEIGHT);
        
        viz_renderer_ = SDL_CreateRenderer(viz_window_, -1, SDL_RENDERER_ACCELERATED);
        if (!viz_renderer_) {
            spdlog::error("Visualization renderer creation failed: {}", SDL_GetError());
            return false;
        }
        
        // Enable alpha blending for visualization renderer
        SDL_SetRenderDrawBlendMode(viz_renderer_, SDL_BLENDMODE_BLEND);
        
        spdlog::debug("Visualization renderer created successfully");
        
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
                                     GRID_SIZE, GRID_SIZE, 0.3f, 0.3f, 12345);
        
        spdlog::info("Brain initialized successfully:");
        spdlog::info("  - Addressing: {} neuron bits, {} dendrite bits = {} max neurons", 
                     NEURON_ADDRESS_BITS, DENDRITE_ADDRESS_BITS, MAX_NEURONS);
        spdlog::info("  - Sensor grid: {}x{} = {} sensors", GRID_SIZE, GRID_SIZE, GRID_SIZE * GRID_SIZE);
        spdlog::info("  - Neural network ready for processing");
    }
    
    IsometricPoint project_to_isometric(const Vec3& point) {
        // Side view projection: rotate by -20° around Y for slight angle, then 20° around X for top-down tilt
        float cos_y = 0.940f;  // cos(-20°)
        float sin_y = 0.342f;  // sin(-20°) 
        float cos_x = 0.940f;  // cos(20°)
        float sin_x = 0.342f;  // sin(20°)
        
        // Scale and center the coordinates
        float scale = 250.0f;
        float center_x = VIZ_WINDOW_WIDTH / 2.0f;
        float center_y = VIZ_WINDOW_HEIGHT / 2.0f;
        
        // Apply Y rotation first (horizontal viewing angle) - flip the rotation
        float x1 = point.x * cos_y + point.z * sin_y;  // Changed sign to flip view
        float y1 = point.y;
        float z1 = -point.x * sin_y + point.z * cos_y;  // Changed sign to flip view
        
        // Then apply X rotation (vertical viewing angle)
        float x2 = x1;
        float y2 = y1 * cos_x - z1 * sin_x;
        
        return {center_x + x2 * scale, center_y - y2 * scale};
    }
    
    struct Color {
        uint8_t r, g, b;
    };
    
    Color depth_to_color(float z_position) {
        // Convert Z position (-1.0 to +1.0) to hue (blue to red)
        // Sensors at z=-1.0 (front) = blue (240°)
        // Actuators at z=+1.0 (back) = red (0°)
        float normalized_depth = (z_position + 1.0f) / 2.0f;  // 0.0 to 1.0
        float hue = 240.0f * (1.0f - normalized_depth);  // 240° to 0°
        
        // HSV to RGB conversion with full saturation and value
        float saturation = 1.0f;
        float value = 1.0f;
        
        float c = value * saturation;
        float h_prime = hue / 60.0f;
        float x = c * (1.0f - std::abs(std::fmod(h_prime, 2.0f) - 1.0f));
        
        float r, g, b;
        if (h_prime >= 0.0f && h_prime < 1.0f) {
            r = c; g = x; b = 0.0f;
        } else if (h_prime >= 1.0f && h_prime < 2.0f) {
            r = x; g = c; b = 0.0f;
        } else if (h_prime >= 2.0f && h_prime < 3.0f) {
            r = 0.0f; g = c; b = x;
        } else if (h_prime >= 3.0f && h_prime < 4.0f) {
            r = 0.0f; g = x; b = c;
        } else if (h_prime >= 4.0f && h_prime < 5.0f) {
            r = x; g = 0.0f; b = c;
        } else {
            r = c; g = 0.0f; b = x;
        }
        
        return {
            static_cast<uint8_t>((r) * 255),
            static_cast<uint8_t>((g) * 255),
            static_cast<uint8_t>((b) * 255)
        };
    }
    
    void render_z_chart() {
        constexpr int CHART_WIDTH = 300;
        constexpr int CHART_HEIGHT = 150;
        constexpr int CHART_X = VIZ_WINDOW_WIDTH - CHART_WIDTH - 20;
        constexpr int CHART_Y = 20;
        
        // Draw chart background
        SDL_SetRenderDrawColor(viz_renderer_, 40, 40, 40, 200);
        SDL_Rect chart_bg = {CHART_X - 10, CHART_Y - 10, CHART_WIDTH + 20, CHART_HEIGHT + 20};
        SDL_RenderFillRect(viz_renderer_, &chart_bg);
        
        // Draw chart border
        SDL_SetRenderDrawColor(viz_renderer_, 100, 100, 100, 255);
        SDL_RenderDrawRect(viz_renderer_, &chart_bg);
        
        // Find maximum total activations across all time slots for scaling
        int max_total_activations = 1;
        for (int t = 0; t < CHART_HISTORY_SIZE; ++t) {
            int total_for_time_slot = 0;
            for (int z = 0; z < Z_BINS; ++z) {
                total_for_time_slot += z_distribution_history_[t][z];
            }
            max_total_activations = std::max(max_total_activations, total_for_time_slot);
        }
        
        // Draw time series as stacked bars
        int bar_width = CHART_WIDTH / CHART_HISTORY_SIZE;
        for (int t = 0; t < CHART_HISTORY_SIZE; ++t) {
            int time_index = (current_history_index_ + t + 1) % CHART_HISTORY_SIZE;
            int x = CHART_X + t * bar_width;
            
            int y_offset = 0;
            for (int z = 0; z < Z_BINS; ++z) {
                int count = z_distribution_history_[time_index][z];
                if (count > 0) {
                    // Scale each individual bin relative to the maximum total activations
                    int bar_height = (count * CHART_HEIGHT) / max_total_activations;
                    
                    // Color based on Z position using existing depth_to_color function
                    float z_position = (static_cast<float>(z) / (Z_BINS - 1)) * 2.0f - 1.0f;  // Convert bin to -1.0 to +1.0
                    Color bin_color = depth_to_color(z_position);
                    SDL_SetRenderDrawColor(viz_renderer_, bin_color.r, bin_color.g, bin_color.b, 255);
                    
                    SDL_Rect bar = {x, CHART_Y + CHART_HEIGHT - y_offset - bar_height, 
                                   bar_width - 1, bar_height};
                    SDL_RenderFillRect(viz_renderer_, &bar);
                    
                    y_offset += bar_height;
                }
            }
        }
        
        // Draw sensor activation line graph overlay
        SDL_SetRenderDrawColor(viz_renderer_, 255, 255, 0, 255); // Yellow line
        
        // Find max sensor activations for scaling
        int max_sensor_activations = 1;
        for (int t = 0; t < CHART_HISTORY_SIZE; ++t) {
            max_sensor_activations = std::max(max_sensor_activations, sensor_activation_history_[t]);
        }
        
        // Draw line connecting sensor activation points
        for (int t = 1; t < CHART_HISTORY_SIZE; ++t) {
            int time_index1 = (current_history_index_ + t) % CHART_HISTORY_SIZE;
            int time_index2 = (current_history_index_ + t + 1) % CHART_HISTORY_SIZE;
            
            int x1 = CHART_X + (t - 1) * bar_width + bar_width / 2;
            int x2 = CHART_X + t * bar_width + bar_width / 2;
            
            int y1 = CHART_Y + CHART_HEIGHT - (sensor_activation_history_[time_index1] * CHART_HEIGHT) / max_sensor_activations;
            int y2 = CHART_Y + CHART_HEIGHT - (sensor_activation_history_[time_index2] * CHART_HEIGHT) / max_sensor_activations;
            
            SDL_RenderDrawLine(viz_renderer_, x1, y1, x2, y2);
        }
        
        // Draw max Z coordinate line graph overlay
        SDL_SetRenderDrawColor(viz_renderer_, 255, 0, 255, 255); // Magenta line
        
        // Draw line connecting max Z points (normalized to -1.0 to +1.0 range)
        for (int t = 1; t < CHART_HISTORY_SIZE; ++t) {
            int time_index1 = (current_history_index_ + t) % CHART_HISTORY_SIZE;
            int time_index2 = (current_history_index_ + t + 1) % CHART_HISTORY_SIZE;
            
            float max_z1 = max_z_history_[time_index1];
            float max_z2 = max_z_history_[time_index2];
            
            // Only draw if we have valid data
            if (max_z1 > -2.0f && max_z2 > -2.0f) {
                int x1 = CHART_X + (t - 1) * bar_width + bar_width / 2;
                int x2 = CHART_X + t * bar_width + bar_width / 2;
                
                // Normalize Z coordinates from [-1.0, +1.0] to [0, CHART_HEIGHT]
                int y1 = CHART_Y + CHART_HEIGHT - static_cast<int>((max_z1 + 1.0f) * CHART_HEIGHT / 2.0f);
                int y2 = CHART_Y + CHART_HEIGHT - static_cast<int>((max_z2 + 1.0f) * CHART_HEIGHT / 2.0f);
                
                SDL_RenderDrawLine(viz_renderer_, x1, y1, x2, y2);
            }
        }
        
        // Draw labels
        SDL_SetRenderDrawColor(viz_renderer_, 200, 200, 200, 255);
        // Title area (just draw a simple line for now)
        SDL_RenderDrawLine(viz_renderer_, CHART_X, CHART_Y - 5, CHART_X + CHART_WIDTH, CHART_Y - 5);
    }
    
    void update_z_chart() {
        // Move to next time slot
        current_history_index_ = (current_history_index_ + 1) % CHART_HISTORY_SIZE;
        
        // Clear current time slot
        std::fill(z_distribution_history_[current_history_index_].begin(), 
                 z_distribution_history_[current_history_index_].end(), 0);
        sensor_activation_history_[current_history_index_] = 0;
        max_z_history_[current_history_index_] = -2.0f;  // Reset to invalid value
    }
    
    void update_firing_bins() {
        // Shift all bins down by one (oldest bin is discarded)
        for (int i = FIRING_TIME_BINS - 1; i > 0; --i) {
            firing_time_bins_[i] = std::move(firing_time_bins_[i - 1]);
        }
        
        // Clear the newest bin (index 0)
        firing_time_bins_[0].clear();
        current_firing_bin_ = 0;  // Always add to bin 0 (newest)
    }
    
    void update_ripples() {
        uint32_t current_timestamp = simulation_timestamp_.load();
        constexpr uint32_t RIPPLE_DURATION = 4;  // simulation steps to complete
        
        // Update existing ripples and activate grid cells
        for (auto& ripple : active_ripples_) {
            uint32_t age = current_timestamp - ripple.start_time;
            float new_radius = (static_cast<float>(age) / RIPPLE_DURATION) * ripple.max_radius;
            
            // Only activate cells at the current expanding edge
            if (new_radius > ripple.current_radius) {
                int min_radius = static_cast<int>(ripple.current_radius);
                int max_radius = static_cast<int>(new_radius) + 1;
                
                // Activate cells in the expanding ring
                for (int dy = -max_radius; dy <= max_radius; ++dy) {
                    for (int dx = -max_radius; dx <= max_radius; ++dx) {
                        float distance = std::sqrt(dx * dx + dy * dy);
                        
                        // Only activate cells in the current ring
                        if (distance > min_radius && distance <= max_radius) {
                            int cell_x = ripple.center_x + dx;
                            int cell_y = ripple.center_y + dy;
                            
                            if (cell_x >= 0 && cell_x < GRID_SIZE && cell_y >= 0 && cell_y < GRID_SIZE) {
                                auto& cell = grid_[cell_y][cell_x];
                                
                                if (ripple.source_type == CellSource::USER) {
                                    cell.user_intensity = 0.5f;  // Dimmer than direct click
                                    cell.last_source = CellSource::USER;
                                } else {
                                    cell.actuator_intensity = 0.5f;  // Dimmer than direct actuator
                                    cell.last_source = CellSource::ACTUATOR;
                                }
                                cell.activation_sent = false;  // Allow new activation
                            }
                        }
                    }
                }
                
                ripple.current_radius = new_radius;
            }
        }
        
        // Remove completed ripples
        active_ripples_.erase(
            std::remove_if(active_ripples_.begin(), active_ripples_.end(),
                [current_timestamp](const RippleEffect& ripple) {
                    return (current_timestamp - ripple.start_time) > RIPPLE_DURATION;
                }), 
            active_ripples_.end()
        );
    }
    
    bool all_shards_completed() {
        for (const auto& completed : shard_completed_) {
            if (!completed.load()) {
                return false;
            }
        }
        return true;
    }
    
    void advance_timestamp() {
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
        
        // Update charts synchronized with simulation timestamps
        update_z_chart();
        update_firing_bins();        
        
        last_timestamp_time_ = current_time;
    }
    
    void process_scheduled_activations() {
        uint32_t current_timestamp = simulation_timestamp_.load();
        
        if (current_timestamp >= next_scheduled_activation_) {
            // Calculate position on circle
            float center_x = GRID_SIZE / 2.0f;
            float center_y = GRID_SIZE / 2.0f;
            float radius = GRID_SIZE / 4.0f;  // Half the grid size radius
            
            // Calculate angle for current point (clockwise, starting from top)
            float angle = (static_cast<float>(current_schedule_point_) / SCHEDULE_POINTS) * 2.0f * M_PI - M_PI_2;
            
            // Calculate grid position
            int grid_x = static_cast<int>(center_x + radius * std::cos(angle));
            int grid_y = static_cast<int>(center_y + radius * std::sin(angle));
            
            // Clamp to grid bounds
            grid_x = std::max(0, std::min(GRID_SIZE - 1, grid_x));
            grid_y = std::max(0, std::min(GRID_SIZE - 1, grid_y));
            
            // Activate the cell
            auto& cell = grid_[grid_y][grid_x];
            cell.user_intensity = 1.0f;
            cell.last_source = CellSource::USER;
            cell.activation_sent = false;
            
            // Create ripple effect
            //active_ripples_.emplace_back(grid_x, grid_y, CellSource::USER, current_timestamp);
            
            spdlog::info("Scheduled activation at point {}/12: grid=({}, {}), angle={:.1f}°", 
                        current_schedule_point_ + 1, grid_x, grid_y, angle * 180.0f / M_PI);
            
            // Move to next point
            current_schedule_point_ = (current_schedule_point_ + 1) % SCHEDULE_POINTS;
            next_scheduled_activation_ = current_timestamp + SCHEDULE_INTERVAL;
        }
    }
    
    void initialize_shard_threads() {
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
                //spdlog::info("Shard {} completed timestamp {}", shard_idx, current_timestamp);
            } else {
                // Small sleep to prevent excessive CPU usage when waiting
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
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
            render_visualization();
            
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
            grid_[grid_y][grid_x].activation_sent = false;  // Reset to allow new activation
            
            // Create ripple effect
            //active_ripples_.emplace_back(grid_x, grid_y, CellSource::USER, simulation_timestamp_.load());
            
            spdlog::debug("User activated cell ({}, {}) at screen pos ({}, {})", grid_x, grid_y, x, y);
        }
    }
    
    void update() {
        // Wait for all shards to complete current timestamp before advancing
        if (all_shards_completed()) {
            simulation_step();
            advance_timestamp();
            spdlog::debug("Simulation step completed, timestamp: {}", simulation_timestamp_.load());
        } else {
            // sleep briefly to avoid busy-waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    void simulation_step() {
        uint32_t current_timestamp = simulation_timestamp_.load();
        
        // 1. Fade all cells by 1/4
        fade_grid();
        
        // 2. Generate sensor activations for all grid elements
        generate_sensor_activations();
        
        // 3. Handle actuator outputs
        process_actuator_outputs();
        
        // 4. Update ripple effects
        update_ripples();
        
        // 5. Process scheduled activations
        //process_scheduled_activations();
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
                    cell.activation_sent = false;  // Reset when cell becomes inactive
                }
            }
        }
    }
    
    void generate_sensor_activations() {
        std::vector<SensorActivation> activations;
        
        // Generate sensor input once for any non-black cell that hasn't sent activation yet
        for (int y = 0; y < GRID_SIZE; ++y) {
            for (int x = 0; x < GRID_SIZE; ++x) {
                auto& cell = grid_[y][x];
                
                // Send activation for any active cell that hasn't sent one yet
                if (!cell.activation_sent && cell.get_total_intensity() > 0.01f) {
                    uint32_t sensor_index = y * GRID_SIZE + x;
                    uint8_t mode_bitmap = 0;
                    float intensity = cell.get_total_intensity();
                    
                    if (cell.user_intensity > 0.01f) {
                        // User input: use modes 0-3 based on intensity
                        if (intensity > 0.75f) mode_bitmap |= (1 << 0);  // Mode 0: Very bright
                        else if (intensity > 0.5f) mode_bitmap |= (1 << 1);   // Mode 1: Bright
                        else if (intensity > 0.25f) mode_bitmap |= (1 << 2);  // Mode 2: Medium
                        else mode_bitmap |= (1 << 3);                         // Mode 3: Dim
                    } else if (cell.actuator_intensity > 0.01f) {
                        // Actuator feedback: use all 4 modes to indicate self-generated
                        mode_bitmap = 0xF; // All 4 modes active
                    }
                    
                    if (mode_bitmap != 0) {
                        activations.emplace_back(sensor_index, mode_bitmap, intensity);
                        cell.activation_sent = true;  // Mark as sent to prevent repeats
                    }
                }
            }
        }
        
        // Process sensor activations and send to neural network
        uint32_t current_timestamp = simulation_timestamp_.load();
        auto targeted_activations = process_sensor_activations(brain_->sensor_grid, activations, current_timestamp);
        
        // Track sensor activation count for chart (both raw and targeted)
        sensor_activation_history_[current_history_index_] += static_cast<int>(activations.size());
        
        if (!targeted_activations.empty()) {
            spdlog::debug("Generated {} sensor activations -> {} targeted activations", 
                         activations.size(), targeted_activations.size());
            message_processor_.send_activations_to_shards(targeted_activations);
        } else if (!activations.empty()) {
            spdlog::debug("Generated {} sensor activations but no targeted activations", activations.size());
        }
    }
        
    void process_actuator_outputs() {
        // Get all actuation events
        auto actuation_events = brain_->actuation_queue.pop_all();
        
        if (!actuation_events.empty()) {
            spdlog::info("Processing {} actuator outputs", actuation_events.size());
        }
        
        for (const auto& event : actuation_events) {
            spdlog::info("Actuator fired: neuron_pos=({:.3f}, {:.3f}, {:.3f}) timestamp={}", 
                        event.position.x, event.position.y, event.position.z, event.timestamp);
            
            // Convert world position to grid coordinates
            // Assuming the sensor grid spans the same area as the flow field
            float norm_x = (event.position.x - (-1.0f)) / (1.0f - (-1.0f));  // Normalize to 0-1
            float norm_y = (event.position.y - (-1.0f)) / (1.0f - (-1.0f));  // Normalize to 0-1
            
            int grid_x = static_cast<int>(norm_x * GRID_SIZE);
            int grid_y = static_cast<int>(norm_y * GRID_SIZE);
            
            // Clamp to grid bounds
            int clamped_grid_x = std::max(0, std::min(GRID_SIZE - 1, grid_x));
            int clamped_grid_y = std::max(0, std::min(GRID_SIZE - 1, grid_y));
            
            spdlog::info("Actuator mapping: world=({:.3f}, {:.3f}) -> norm=({:.3f}, {:.3f}) -> grid=({}, {}) -> clamped=({}, {})", 
                        event.position.x, event.position.y, norm_x, norm_y, 
                        grid_x, grid_y, clamped_grid_x, clamped_grid_y);
            
            // Set actuator intensity
            grid_[clamped_grid_y][clamped_grid_x].actuator_intensity = 1.0f;
            grid_[clamped_grid_y][clamped_grid_x].last_source = CellSource::ACTUATOR;
            grid_[clamped_grid_y][clamped_grid_x].activation_sent = false;  // Reset to allow new activation
            
            // Create ripple effect
            //active_ripples_.emplace_back(clamped_grid_x, clamped_grid_y, CellSource::ACTUATOR, simulation_timestamp_.load());
        }
    }
    
    void render_visualization() {
        // Clear visualization window to black
        SDL_SetRenderDrawColor(viz_renderer_, 0, 0, 0, 255);
        SDL_RenderClear(viz_renderer_);
                
        // Move recent firings to local for processing, update chart, and store in time bins
        {
            std::lock_guard<std::mutex> lock(firing_mutex_);
            for (const auto& event : recent_firings_) {                
                // Add to current Z-coordinate distribution
                float z_norm = (event.position.z + 1.0f) / 2.0f;  // 0.0 to 1.0
                int bin = static_cast<int>(z_norm * Z_BINS);
                bin = std::max(0, std::min(Z_BINS - 1, bin));
                z_distribution_history_[current_history_index_][bin]++;
                
                // Store position in current firing time bin
                firing_time_bins_[current_firing_bin_].push_back(event.position);
                
                // Track max Z coordinate for this time slot
                max_z_history_[current_history_index_] = std::max(max_z_history_[current_history_index_], event.position.z);
            }
            recent_firings_.clear();
        }
                
        // Draw sensors (at bottom of space)
        SDL_SetRenderDrawColor(viz_renderer_, 0, 100, 255, 255); // Blue for sensors
        for (uint32_t sensor_idx = 0; sensor_idx < GRID_SIZE * GRID_SIZE; ++sensor_idx) {
            const Sensor& sensor = brain_->sensor_grid.sensors[sensor_idx];
            IsometricPoint iso_point = project_to_isometric(sensor.position);
            
            SDL_Rect sensor_rect;
            sensor_rect.x = static_cast<int>(iso_point.x) - 2;
            sensor_rect.y = static_cast<int>(iso_point.y) - 2;
            sensor_rect.w = 4;
            sensor_rect.h = 4;
            SDL_RenderFillRect(viz_renderer_, &sensor_rect);
        }
        
        // Draw actuator neurons
        SDL_SetRenderDrawColor(viz_renderer_, 255, 0, 0, 255); // Red for actuators
        for (uint32_t neuron_idx = 0; neuron_idx < MAX_NEURONS; ++neuron_idx) {
            if (brain_->neurons[neuron_idx].is_actuator) {
                IsometricPoint iso_point = project_to_isometric(brain_->neurons[neuron_idx].position);
                
                SDL_Rect actuator_rect;
                actuator_rect.x = static_cast<int>(iso_point.x) - 3;
                actuator_rect.y = static_cast<int>(iso_point.y) - 3;
                actuator_rect.w = 6;
                actuator_rect.h = 6;
                SDL_RenderFillRect(viz_renderer_, &actuator_rect);
            }
        }
        
        // Draw regular neurons (small dark gray dots)
        SDL_SetRenderDrawColor(viz_renderer_, 80, 80, 80, 255); // Dark gray for neurons
        for (uint32_t neuron_idx = 0; neuron_idx < MAX_NEURONS; ++neuron_idx) {
            if (!brain_->neurons[neuron_idx].is_actuator) {
                IsometricPoint iso_point = project_to_isometric(brain_->neurons[neuron_idx].position);
                
                SDL_Rect neuron_rect;
                neuron_rect.x = static_cast<int>(iso_point.x) - 1;
                neuron_rect.y = static_cast<int>(iso_point.y) - 1;
                neuron_rect.w = 2;
                neuron_rect.h = 2;
                SDL_RenderFillRect(viz_renderer_, &neuron_rect);
            }
        }
        
        // Draw neuron firings from time bins with fading effect
        for (int bin_index = 0; bin_index < FIRING_TIME_BINS; ++bin_index) {
            const auto& positions = firing_time_bins_[bin_index];
            
            // Calculate alpha based on age (bin 0 = newest, bin 9 = oldest)
            float alpha = 1.0f - (static_cast<float>(bin_index) / FIRING_TIME_BINS);
            if (alpha <= 0.0f) continue;
            
            for (const auto& position : positions) {
                IsometricPoint iso_point = project_to_isometric(position);
                
                // Get depth-based color (blue for sensors, red for actuators)
                Color firing_color = depth_to_color(position.z);
                SDL_SetRenderDrawColor(viz_renderer_, firing_color.r, firing_color.g, firing_color.b, static_cast<uint8_t>(alpha * 255));
                
                int radius = 5;
                for (int y = -radius; y <= radius; ++y) {
                    for (int x = -radius; x <= radius; ++x) {
                        if (x*x + y*y <= radius*radius) {
                            SDL_RenderDrawPoint(viz_renderer_, 
                                               static_cast<int>(iso_point.x) + x,
                                               static_cast<int>(iso_point.y) + y);
                        }
                    }
                }
            }
        }
        
        // Draw the Z-coordinate chart
        render_z_chart();
        
        SDL_RenderPresent(viz_renderer_);
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
        if (viz_renderer_) {
            SDL_DestroyRenderer(viz_renderer_);
            viz_renderer_ = nullptr;
        }
        
        if (viz_window_) {
            SDL_DestroyWindow(viz_window_);
            viz_window_ = nullptr;
        }
        
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