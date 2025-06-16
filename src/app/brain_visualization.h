#ifndef BRAIN_VISUALIZATION_H
#define BRAIN_VISUALIZATION_H

#include "constants.h"
#include "visualization.h"
#include "neural_simulation.h"
#include <SDL2/SDL.h>
#include <vector>

using namespace neuronlib;

class BrainVisualization {
private:
    SDL_Window* viz_window_;
    SDL_Renderer* viz_renderer_;
    
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

public:
    BrainVisualization();
    ~BrainVisualization();
    
    bool initialize();
    void render(NeuralSimulation& neural_sim);
    void update_charts();
    void process_firing_events(const std::vector<NeuronFiringEvent>& events);
    void track_sensor_activations(int count);
    void cleanup();

private:
    void render_z_chart();
    void update_z_chart();
    void update_firing_bins();
};

#endif