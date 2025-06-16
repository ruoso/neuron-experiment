#include "neuron_grid_experiment.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <iostream>
#include <cmath>
#include <future>
#include <queue>
#include <condition_variable>
#include <functional>
#include <pthread.h>
#include <sched.h>
#include <cstring>
#include <algorithm>

NeuronGridExperiment::NeuronGridExperiment() 
    : window_(nullptr), renderer_(nullptr),
      running_(false), grid_(GRID_SIZE, std::vector<GridCell>(GRID_SIZE)) {
    
    // Initialize logging
    initialize_logging();
    
    // Initialize neural simulation
    neural_sim_.initialize();
    
    // Set up firing callback for visualization
    neural_sim_.set_firing_callback([this](const std::vector<NeuronFiringEvent>& events) {
        // This callback is called from the neural simulation for any additional processing
        // The neural simulation already handles the basic firing event collection
    });
    
    // Start neural simulation threads
    neural_sim_.start();
    
    last_update_ = std::chrono::steady_clock::now();
}

NeuronGridExperiment::~NeuronGridExperiment() {
    neural_sim_.stop();
    cleanup();
}

void NeuronGridExperiment::initialize_logging() {
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

bool NeuronGridExperiment::initialize() {
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
    
    // Initialize brain visualization
    if (!brain_viz_.initialize()) {
        spdlog::error("Brain visualization initialization failed");
        return false;
    }
    
    spdlog::info("Application initialization complete");
    return true;
}


void NeuronGridExperiment::update_ripples() {
    uint32_t current_timestamp = neural_sim_.get_current_timestamp();
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



void NeuronGridExperiment::run() {
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
    
}

void NeuronGridExperiment::handle_events() {
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

void NeuronGridExperiment::handle_mouse_click(int x, int y) {
    // Convert screen coordinates to grid coordinates
    int grid_x = (x - GRID_OFFSET_X) / CELL_SIZE;
    int grid_y = (y - GRID_OFFSET_Y) / CELL_SIZE;
    
    if (grid_x >= 0 && grid_x < GRID_SIZE && grid_y >= 0 && grid_y < GRID_SIZE) {
        grid_[grid_y][grid_x].user_intensity = 1.0f;
        grid_[grid_y][grid_x].last_source = CellSource::USER;
        grid_[grid_y][grid_x].activation_sent = false;  // Reset to allow new activation
        
        // Create ripple effect
        //active_ripples_.emplace_back(grid_x, grid_y, CellSource::USER, neural_sim_.get_current_timestamp());
        
        spdlog::debug("User activated cell ({}, {}) at screen pos ({}, {})", grid_x, grid_y, x, y);
    }
}

void NeuronGridExperiment::update() {
    // Wait for neural simulation to be ready, then advance
    if (neural_sim_.is_ready_to_advance()) {
        simulation_step();
        neural_sim_.advance_timestamp();
        
        // Update charts synchronized with simulation timestamps
        brain_viz_.update_charts();
        
        spdlog::debug("Simulation step completed, timestamp: {}", neural_sim_.get_current_timestamp());
    } else {
        // sleep briefly to avoid busy-waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void NeuronGridExperiment::simulation_step() {
    uint32_t current_timestamp = neural_sim_.get_current_timestamp();
    
    // 1. Fade all cells by 1/4
    fade_grid();
    
    // 2. Generate sensor activations for all grid elements
    generate_sensor_activations();
    
    // 3. Handle actuator outputs
    process_actuator_outputs();
    
    // 4. Update ripple effects
    update_ripples();
    
}

void NeuronGridExperiment::fade_grid() {
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

void NeuronGridExperiment::generate_sensor_activations() {
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
    uint32_t current_timestamp = neural_sim_.get_current_timestamp();
    auto targeted_activations = process_sensor_activations(neural_sim_.get_brain().sensor_grid, activations, current_timestamp);
    
    // Track sensor activation count for chart (both raw and targeted)
    brain_viz_.track_sensor_activations(static_cast<int>(activations.size()));
    
    if (!targeted_activations.empty()) {
        spdlog::debug("Generated {} sensor activations -> {} targeted activations", 
                     activations.size(), targeted_activations.size());
        neural_sim_.send_sensor_activations(targeted_activations);
    } else if (!activations.empty()) {
        spdlog::debug("Generated {} sensor activations but no targeted activations", activations.size());
    }
}
    
void NeuronGridExperiment::process_actuator_outputs() {
    // Get all actuation events
    auto actuation_events = neural_sim_.get_actuator_events();
    
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
        //active_ripples_.emplace_back(clamped_grid_x, clamped_grid_y, CellSource::ACTUATOR, neural_sim_.get_current_timestamp());
    }
}

void NeuronGridExperiment::render_visualization() {
    brain_viz_.render(neural_sim_);
}

void NeuronGridExperiment::render() {
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

void NeuronGridExperiment::cleanup() {
    brain_viz_.cleanup();
    
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

int main(int argc, char* argv[]) {
    NeuronGridExperiment app;
    app.run();
    return 0;
}