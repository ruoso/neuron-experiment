#include <SDL2/SDL.h>
#include "spatial_operations.h"
#include "flow_field.h"
#include "sensor.h"
#include "actuator.h"
#include "activation.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

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
    
    // Timing
    std::chrono::steady_clock::time_point last_update_;
    uint32_t simulation_timestamp_;
    
public:
    NeuronExperimentApp() : window_(nullptr), renderer_(nullptr), running_(false), 
                           grid_(GRID_SIZE, std::vector<GridCell>(GRID_SIZE)),
                           message_processor_(10),
                           simulation_timestamp_(0) {
        
        // Initialize neural network
        initialize_brain();
        
        last_update_ = std::chrono::steady_clock::now();
    }
    
    ~NeuronExperimentApp() {
        cleanup();
    }
    
    bool initialize() {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
            return false;
        }
        
        window_ = SDL_CreateWindow("Neuron Experiment",
                                 SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                 WINDOW_WIDTH, WINDOW_HEIGHT,
                                 SDL_WINDOW_SHOWN);
        
        if (!window_) {
            std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
            return false;
        }
        
        renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED);
        if (!renderer_) {
            std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
            return false;
        }
        
        return true;
    }
    
    void initialize_brain() {
        // Create a simple 3D flow field
        FlowField3D flow_field(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.4f, 0.4f, 0.1f);
        
        // Create brain with matching sensor grid
        brain_ = populate_neuron_grid(flow_field, 1.0f, 45.0f, 0.1f, 0.5f,
                                     GRID_SIZE, GRID_SIZE, 0.3f, 0.1f, 12345);
        
        std::cout << "Brain initialized with " << GRID_SIZE << "x" << GRID_SIZE << " sensor grid" << std::endl;
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
        }
    }
    
    void update() {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_update_);
        
        if (elapsed.count() >= 100) {
            simulation_step();
            last_update_ = current_time;
            simulation_timestamp_++;
        }
    }
    
    void simulation_step() {
        // 1. Fade all cells by 1/4
        fade_grid();
        
        // 2. Generate sensor activations for non-black cells
        generate_sensor_activations();
        
        // 3. Process neural network for one step
        process_neural_network();
        
        // 4. Handle actuator outputs
        process_actuator_outputs();
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
                    }
                    
                    if (cell.actuator_intensity > 0.01f) {
                        // Actuator feedback: use all 4 modes to indicate self-generated
                        mode_bitmap = 0xF; // All 4 modes active
                    }
                    
                    if (mode_bitmap != 0) {
                        activations.emplace_back(sensor_index, mode_bitmap, cell.get_total_intensity());
                    }
                }
            }
        }
        
        // Process sensor activations and send to neural network
        auto targeted_activations = process_sensor_activations(brain_->sensor_grid, activations, simulation_timestamp_);
        message_processor_.send_activations_to_shards(targeted_activations);
    }
    
    void process_neural_network() {
        // Process one tick for each shard
        for (uint32_t shard_idx = 0; shard_idx < NUM_ACTIVATION_SHARDS; ++shard_idx) {
            auto& shard = message_processor_.get_shard(shard_idx);
            shard.process_tick(*brain_, simulation_timestamp_, &message_processor_);
        }
    }
    
    void process_actuator_outputs() {
        // Get all actuation events
        auto actuation_events = brain_->actuation_queue.pop_all();
        
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
            
            // Set actuator intensity
            grid_[grid_y][grid_x].actuator_intensity = 1.0f;
            grid_[grid_y][grid_x].last_source = CellSource::ACTUATOR;
        }
        
        if (!actuation_events.empty()) {
            std::cout << "Processed " << actuation_events.size() << " actuation events" << std::endl;
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