#include "brain_visualization.h"
#include <spdlog/spdlog.h>
#include <algorithm>

BrainVisualization::BrainVisualization()
    : viz_window_(nullptr), viz_renderer_(nullptr),
      z_distribution_history_(CHART_HISTORY_SIZE, std::vector<int>(Z_BINS, 0)),
      sensor_activation_history_(CHART_HISTORY_SIZE, 0),
      max_z_history_(CHART_HISTORY_SIZE, -2.0f),
      current_history_index_(0),
      firing_time_bins_(FIRING_TIME_BINS),
      current_firing_bin_(0) {
}

BrainVisualization::~BrainVisualization() {
    cleanup();
}

bool BrainVisualization::initialize() {
    SPDLOG_INFO("Initializing brain visualization...");
    
    // Create visualization window
    viz_window_ = SDL_CreateWindow("Neuron Experiment - 3D Visualization",
                                 WINDOW_WIDTH + 50, SDL_WINDOWPOS_UNDEFINED,
                                 VIZ_WINDOW_WIDTH, VIZ_WINDOW_HEIGHT,
                                 SDL_WINDOW_SHOWN);
    
    if (!viz_window_) {
        spdlog::error("Visualization window creation failed: {}", SDL_GetError());
        return false;
    }
    SPDLOG_DEBUG("Visualization window created: {}x{}", VIZ_WINDOW_WIDTH, VIZ_WINDOW_HEIGHT);
    
    viz_renderer_ = SDL_CreateRenderer(viz_window_, -1, SDL_RENDERER_ACCELERATED);
    if (!viz_renderer_) {
        spdlog::error("Visualization renderer creation failed: {}", SDL_GetError());
        return false;
    }
    
    // Enable alpha blending for visualization renderer
    SDL_SetRenderDrawBlendMode(viz_renderer_, SDL_BLENDMODE_BLEND);
    
    SPDLOG_DEBUG("Visualization renderer created successfully");
    SPDLOG_INFO("Brain visualization initialization complete");
    return true;
}

void BrainVisualization::render(NeuralSimulation& neural_sim) {
    // Clear visualization window to black
    SDL_SetRenderDrawColor(viz_renderer_, 0, 0, 0, 255);
    SDL_RenderClear(viz_renderer_);
            
    // Get and process recent firings
    auto recent_firings = neural_sim.get_recent_firings();
    process_firing_events(recent_firings);
            
    const Brain& brain = neural_sim.get_brain();
    
    // Draw sensors (at bottom of space)
    SDL_SetRenderDrawColor(viz_renderer_, 0, 100, 255, 255); // Blue for sensors
    for (uint32_t sensor_idx = 0; sensor_idx < GRID_SIZE * GRID_SIZE; ++sensor_idx) {
        const Sensor& sensor = brain.sensor_grid.sensors[sensor_idx];
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
        if (brain.neurons[neuron_idx].is_actuator) {
            IsometricPoint iso_point = project_to_isometric(brain.neurons[neuron_idx].position);
            
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
        if (!brain.neurons[neuron_idx].is_actuator) {
            IsometricPoint iso_point = project_to_isometric(brain.neurons[neuron_idx].position);
            
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

void BrainVisualization::update_charts() {
    update_z_chart();
    update_firing_bins();
}

void BrainVisualization::process_firing_events(const std::vector<NeuronFiringEvent>& events) {
    for (const auto& event : events) {                
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
}

void BrainVisualization::track_sensor_activations(int count) {
    sensor_activation_history_[current_history_index_] += count;
}

void BrainVisualization::render_z_chart() {
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

void BrainVisualization::update_z_chart() {
    // Move to next time slot
    current_history_index_ = (current_history_index_ + 1) % CHART_HISTORY_SIZE;
    
    // Clear current time slot
    std::fill(z_distribution_history_[current_history_index_].begin(), 
             z_distribution_history_[current_history_index_].end(), 0);
    sensor_activation_history_[current_history_index_] = 0;
    max_z_history_[current_history_index_] = -2.0f;  // Reset to invalid value
}

void BrainVisualization::update_firing_bins() {
    // Shift all bins down by one (oldest bin is discarded)
    for (int i = FIRING_TIME_BINS - 1; i > 0; --i) {
        firing_time_bins_[i] = std::move(firing_time_bins_[i - 1]);
    }
    
    // Clear the newest bin (index 0)
    firing_time_bins_[0].clear();
    current_firing_bin_ = 0;  // Always add to bin 0 (newest)
}

void BrainVisualization::cleanup() {
    if (viz_renderer_) {
        SDL_DestroyRenderer(viz_renderer_);
        viz_renderer_ = nullptr;
    }
    
    if (viz_window_) {
        SDL_DestroyWindow(viz_window_);
        viz_window_ = nullptr;
    }
}