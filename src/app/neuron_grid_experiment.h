#ifndef NEURON_GRID_EXPERIMENT_H
#define NEURON_GRID_EXPERIMENT_H

#include "constants.h"
#include "types.h"
#include "neural_simulation.h"
#include "brain_visualization.h"
#include <SDL2/SDL.h>
#include <vector>
#include <chrono>

using namespace neuronlib;

class NeuronGridExperiment {
private:
    SDL_Window* window_;
    SDL_Renderer* renderer_;
    bool running_;
    
    // Grid state
    std::vector<std::vector<GridCell>> grid_;
    
    // Neural simulation
    NeuralSimulation neural_sim_;
    
    // Brain visualization
    BrainVisualization brain_viz_;
    
    // Timing
    std::chrono::steady_clock::time_point last_update_;
    
    // Ripple effects
    std::vector<RippleEffect> active_ripples_;
    

public:
    NeuronGridExperiment();
    ~NeuronGridExperiment();
    
    bool initialize();
    void run();
    
private:
    void initialize_logging();
    
    void handle_events();
    void handle_mouse_click(int x, int y);
    void update();
    void simulation_step();
    void fade_grid();
    void generate_sensor_activations();
    void process_actuator_outputs();
    
    void update_ripples();
    
    void render();
    void render_visualization();
    
    void cleanup();
};

#endif