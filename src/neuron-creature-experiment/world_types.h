#ifndef NEURON_CREATURE_EXPERIMENT_WORLD_TYPES_H
#define NEURON_CREATURE_EXPERIMENT_WORLD_TYPES_H

#include <cstdint>
#include <vector>
#include <cmath>

namespace neuron_creature_experiment {

struct Vec2 {
    float x, y;
    
    Vec2() : x(0.0f), y(0.0f) {}
    Vec2(float x_val, float y_val) : x(x_val), y(y_val) {}
    
    Vec2 operator+(const Vec2& other) const {
        return Vec2(x + other.x, y + other.y);
    }
    
    Vec2 operator-(const Vec2& other) const {
        return Vec2(x - other.x, y - other.y);
    }
    
    Vec2 operator*(float scalar) const {
        return Vec2(x * scalar, y * scalar);
    }
    
    float magnitude() const {
        return std::sqrt(x * x + y * y);
    }
    
    Vec2 normalized() const {
        float mag = magnitude();
        return mag > 0.0f ? Vec2(x / mag, y / mag) : Vec2(0.0f, 0.0f);
    }
};

enum class TreeLifecycleState {
    SEEDLING,
    MATURE,
    FRUITING,
    DORMANT
};

struct TreeState {
    TreeLifecycleState lifecycle_state;
    float age;
    float state_timer;
    uint32_t fruits_produced;
    
    TreeState() : lifecycle_state(TreeLifecycleState::SEEDLING), age(0.0f), 
                  state_timer(0.0f), fruits_produced(0) {}
};

struct Color {
    float r, g, b;
    
    Color() : r(0.0f), g(0.0f), b(0.0f) {}
    Color(float red, float green, float blue) : r(red), g(green), b(blue) {}
};

struct Tree {
    Vec2 position;
    TreeState state;
    uint32_t tree_id;
    Color color;
    float radius;
    
    Tree() : position(), state(), tree_id(0), color(), radius(3.0f) {}
    Tree(const Vec2& pos, uint32_t id) : position(pos), state(), tree_id(id), color(), radius(3.0f) {}
    
    void update_color_for_state();
};

struct Fruit {
    Vec2 position;
    float maturity;
    float satiation_value;
    bool available;
    uint32_t fruit_id;
    uint32_t parent_tree_id;
    Color color;
    float radius;
    
    Fruit() : position(), maturity(0.0f), satiation_value(0.0f), 
              available(true), fruit_id(0), parent_tree_id(0), color(), radius(1.0f) {}
    
    Fruit(const Vec2& pos, uint32_t id, uint32_t tree_id) 
        : position(pos), maturity(0.0f), satiation_value(0.0f), 
          available(true), fruit_id(id), parent_tree_id(tree_id), color(), radius(1.0f) {}
    
    void update_satiation(float max_satiation) {
        satiation_value = maturity * max_satiation;
    }
    
    void update_color_for_maturity();
};

struct CreatureState {
    Vec2 position;
    float orientation;
    Vec2 velocity;
    float angular_velocity;
    float hunger;
    float energy;
    
    CreatureState() : position(), orientation(0.0f), velocity(), 
                      angular_velocity(0.0f), hunger(0.0f), energy(1.0f) {}
};

struct MotorOutput {
    float left_force;
    float right_force;
    bool eat_action;
    
    MotorOutput() : left_force(0.0f), right_force(0.0f), eat_action(false) {}
    MotorOutput(float left, float right, bool eat = false) 
        : left_force(left), right_force(right), eat_action(eat) {}
};

struct VisionSample {
    Color blended_color;
    float total_intensity;
    float distance;
    
    VisionSample() : blended_color(), total_intensity(0.0f), distance(0.0f) {}
    VisionSample(const Color& color, float intensity, float dist) 
        : blended_color(color), total_intensity(intensity), distance(dist) {}
};

struct SensorData {
    std::vector<VisionSample> vision_samples;
    float forward_velocity;
    float turn_rate;
    float hunger_level;
    float last_satiation;
    
    SensorData() : forward_velocity(0.0f), turn_rate(0.0f), 
                   hunger_level(0.0f), last_satiation(0.0f) {}
};

struct WorldConfig {
    float width;
    float height;
    float simulation_dt;
    uint32_t max_trees;
    uint32_t max_fruits;
    float fruit_max_satiation;
    float creature_eat_radius;
    float hunger_increase_rate;
    
    WorldConfig() : width(100.0f), height(100.0f), simulation_dt(1.0f/60.0f),
                    max_trees(50), max_fruits(200), fruit_max_satiation(10.0f),
                    creature_eat_radius(2.0f), hunger_increase_rate(0.1f) {}
};

} // namespace neuron_creature_experiment

#endif // NEURON_CREATURE_EXPERIMENT_WORLD_TYPES_H