#ifndef NEURON_CREATURE_EXPERIMENT_H
#define NEURON_CREATURE_EXPERIMENT_H

#include "world.h"
#include "creature.h"
#include <SDL2/SDL.h>
#include <chrono>
#include <memory>

namespace neuron_creature_experiment {

class CreatureExperiment {
private:
    SDL_Window* window_;
    SDL_Renderer* renderer_;
    bool running_;
    
    std::unique_ptr<World> world_;
    std::unique_ptr<Creature> creature_;
    
    static constexpr int WINDOW_WIDTH = 1200;
    static constexpr int WINDOW_HEIGHT = 800;
    static constexpr float PIXELS_PER_UNIT = 8.0f;
    static constexpr float CAMERA_FOLLOW_SPEED = 0.1f;
    
    Vec2 camera_position_;
    uint32_t simulation_tick_;
    std::chrono::steady_clock::time_point last_update_;
    
    bool show_debug_info_;
    bool paused_;
    
    float left_motor_activation_;
    float right_motor_activation_;

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
    
    void get_tree_color(const Tree& tree, uint8_t& r, uint8_t& g, uint8_t& b) const;
    void get_fruit_color(const Fruit& fruit, uint8_t& r, uint8_t& g, uint8_t& b) const;
    
    void cleanup();
};

} // namespace neuron_creature_experiment

#endif // NEURON_CREATURE_EXPERIMENT_H