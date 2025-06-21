#ifndef NEURON_CREATURE_EXPERIMENT_H
#define NEURON_CREATURE_EXPERIMENT_H

#include "world.h"
#include "creature.h"
#include "neural_simulation.h"
#include "brain_visualization.h"
#include <SDL2/SDL.h>
#include <chrono>
#include <memory>
#include <array>
#include <string>

using namespace neuronlib;

namespace neuron_creature_experiment {

// Neural sensor/actuator counts
constexpr int NUM_VISION_SENSORS = 192;     // 64 strips × 3 colors
constexpr int NUM_MOTOR_ACTUATORS = 16;     // 16 actuators per motor (8 activators + 8 suppressors)

// Legacy grid constants (for compatibility with old mapping code)
constexpr int NEURAL_GRID_SIZE = 32;
constexpr int VISION_SENSORS_PER_STRIP = 3;  // R, G, B
constexpr int MOTOR_REGION_SIZE = 8;
constexpr int LEFT_MOTOR_X = 8;
constexpr int LEFT_MOTOR_Y = 20;
constexpr int RIGHT_MOTOR_X = 16;
constexpr int RIGHT_MOTOR_Y = 20;
constexpr int HUNGER_SENSOR_X = 12;
constexpr int HUNGER_SENSOR_Y = 28;
constexpr int SATIATION_SENSOR_X = 19;
constexpr int SATIATION_SENSOR_Y = 28;

// Actuator tag constants
constexpr uint8_t LEFT_MOTOR_ACTIVATOR_TAG = 0;
constexpr uint8_t LEFT_MOTOR_SUPPRESSOR_TAG = 1;
constexpr uint8_t RIGHT_MOTOR_ACTIVATOR_TAG = 2;
constexpr uint8_t RIGHT_MOTOR_SUPPRESSOR_TAG = 3;

// Sensor tag constants
constexpr uint16_t VISION_SENSOR_TAG_BASE = 1000;  // Vision sensors: 1000-1191 (192 sensors)
constexpr uint16_t HUNGER_SENSOR_TAG = 2000;
constexpr uint16_t SATIATION_SENSOR_TAG = 2001;

// Sensor/Actuator layout structure using Vec3 positions
struct SensorActuatorLayout {
    std::array<Vec3, NUM_VISION_SENSORS> vision_sensors;           // Vision sensor positions
    std::array<Vec3, NUM_MOTOR_ACTUATORS/2> left_motor_activators;  // Left motor activator positions  
    std::array<Vec3, NUM_MOTOR_ACTUATORS/2> left_motor_suppressors; // Left motor suppressor positions
    std::array<Vec3, NUM_MOTOR_ACTUATORS/2> right_motor_activators; // Right motor activator positions
    std::array<Vec3, NUM_MOTOR_ACTUATORS/2> right_motor_suppressors;// Right motor suppressor positions
    Vec3 hunger_sensor;                                             // Hunger sensor position
    Vec3 satiation_sensor;                                         // Satiation sensor position
};

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
    
    // Survival tracking
    uint32_t ticks_survived_;
    float total_distance_moved_;
    Vec2 last_position_;
    
    // Sensor/Actuator layout
    SensorActuatorLayout layout_;
    std::string layout_encoding_;

public:
    CreatureExperiment(const std::string& layout_encoding);
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
    
    // Layout encoding/decoding methods
    void decode_layout(const std::string& base64_encoding);
    std::string get_layout_filename_suffix() const;
    
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