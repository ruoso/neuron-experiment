#ifndef NEURON_CREATURE_EXPERIMENT_H
#define NEURON_CREATURE_EXPERIMENT_H

#include "world.h"
#include "creature.h"
#include "neural_simulation.h"
#include "brain_visualization.h"
#include <SDL2/SDL.h>
#include <chrono>
#include <memory>

using namespace neuronlib;

namespace neuron_creature_experiment {

// Neural grid mapping constants
constexpr int NEURAL_GRID_SIZE = 32;
constexpr int VISION_ROWS = 6;           // Top 6 rows for vision
constexpr int VISION_SENSORS_PER_STRIP = 3; // R, G, B
constexpr int MOTOR_REGION_SIZE = 8;     // 8x8 regions for each motor
constexpr int LEFT_MOTOR_X = 8;          // Left motor region top-left
constexpr int LEFT_MOTOR_Y = 16;
constexpr int RIGHT_MOTOR_X = 16;        // Right motor region top-left  
constexpr int RIGHT_MOTOR_Y = 16;
constexpr int HUNGER_SENSOR_X = 14;      // Single hunger sensor
constexpr int HUNGER_SENSOR_Y = 28;
constexpr int SATIATION_SENSOR_X = 17;   // Single satiation sensor
constexpr int SATIATION_SENSOR_Y = 28;

class CreatureExperiment {
private:
    SDL_Window* window_;
    SDL_Renderer* renderer_;
    bool running_;
    
    std::unique_ptr<World> world_;
    std::unique_ptr<Creature> creature_;
    
    // Neural simulation
    NeuralSimulation neural_sim_;
    BrainVisualization brain_viz_;
    
    static constexpr int WINDOW_WIDTH = 1200;
    static constexpr int WINDOW_HEIGHT = 800;
    static constexpr float PIXELS_PER_UNIT = 8.0f;
    static constexpr float CAMERA_FOLLOW_SPEED = 0.1f;
    
    Vec2 camera_position_;
    uint32_t simulation_tick_;
    std::chrono::steady_clock::time_point last_update_;
    
    bool show_debug_info_;
    bool paused_;
    bool neural_mode_;
    
    float left_motor_activation_;
    float right_motor_activation_;
    
    // Neural integration state
    float left_motor_feedback_;
    float right_motor_feedback_;
    bool left_motor_sent_;
    bool right_motor_sent_;
    uint32_t vision_activation_counter_;
    
    // Motor activation/suppression tracking for visualization
    float left_motor_activators_;
    float left_motor_suppressors_;
    float right_motor_activators_;
    float right_motor_suppressors_;

public:
    CreatureExperiment();
    ~CreatureExperiment();
    
    bool initialize();
    void run();

private:
    void initialize_logging();
    void initialize_world();
    
    void handle_events();
    void handle_keypress(SDL_Keycode key, bool pressed);
    void update();
    void render();
    
    // Neural integration methods
    void generate_sensor_activations();
    void process_actuator_outputs();
    uint32_t map_vision_strip_to_sensor(int strip_index, int color_channel) const;
    uint32_t map_motor_region_to_sensor(bool is_right_motor, int x_offset, int y_offset) const;
    
    void update_camera();
    Vec2 world_to_screen(const Vec2& world_pos) const;
    Vec2 screen_to_world(const Vec2& screen_pos) const;
    
    void render_background();
    void render_trees();
    void render_fruits();
    void render_creature();
    void render_creature_vision();
    void render_sensor_strips();
    void render_debug_info();
    void render_neural_overlay();
    void render_visualization();
    
    void get_tree_color(const Tree& tree, uint8_t& r, uint8_t& g, uint8_t& b) const;
    void get_fruit_color(const Fruit& fruit, uint8_t& r, uint8_t& g, uint8_t& b) const;
    
    void cleanup();
};

} // namespace neuron_creature_experiment

#endif // NEURON_CREATURE_EXPERIMENT_H