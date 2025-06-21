#ifndef NEURON_CREATURE_EXPERIMENT_CREATURE_H
#define NEURON_CREATURE_EXPERIMENT_CREATURE_H

#include "world_types.h"
#include <vector>

namespace neuron_creature_experiment {

class World;

class Creature {
public:
    Creature(const Vec2& initial_position, float initial_orientation);
    ~Creature() = default;

    void update(uint32_t tick, World& world);
    
    void set_motor_output(const MotorOutput& output);
    MotorOutput get_motor_output() const { return motor_output_; }
    
    SensorData get_sensor_data(const World& world) const;
    CreatureState get_state() const { return state_; }
    
    Vec2 get_position() const { return state_.position; }
    float get_orientation() const { return state_.orientation; }
    Vec2 get_velocity() const { return state_.velocity; }
    float get_hunger() const { return state_.hunger; }
    float get_energy() const { return state_.energy; }
    uint32_t get_fruits_eaten() const { return fruits_eaten_; }
    
    void set_position(const Vec2& position) { state_.position = position; }
    void set_orientation(float orientation) { state_.orientation = orientation; }

private:
    void update_locomotion(float dt);
    void update_internal_state(float dt);
    
    std::vector<VisionSample> get_vision_samples(const World& world) const;
    float get_proprioception_forward_velocity() const;
    float get_proprioception_turn_rate() const;
    float get_hunger_level() const;

    CreatureState state_;
    MotorOutput motor_output_;
    
    float vision_fov_;
    uint32_t vision_strips_;
    float vision_range_;
    
    float last_satiation_spike_;
    uint32_t fruits_eaten_;
    uint32_t last_update_tick_;
};

} // namespace neuron_creature_experiment

#endif // NEURON_CREATURE_EXPERIMENT_CREATURE_H