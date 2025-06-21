#include "creature.h"
#include "world.h"
#include <cmath>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace neuron_creature_experiment {

Creature::Creature(const Vec2& initial_position, float initial_orientation)
    : state_(), motor_output_(), vision_fov_(160.0f * M_PI / 180.0f), vision_strips_(8), 
      vision_range_(20.0f), last_satiation_spike_(0.0f), fruits_eaten_(0), last_update_tick_(0) {
    state_.position = initial_position;
    state_.orientation = initial_orientation;
    state_.velocity = Vec2(0.0f, 0.0f);
    state_.angular_velocity = 0.0f;
    state_.hunger = 0.0f;
    state_.energy = 1.0f;
}

void Creature::update(uint32_t tick, World& world) {
    float dt = 1.0f / 60.0f;
    
    if (last_update_tick_ > 0) {
        uint32_t tick_diff = tick - last_update_tick_;
        dt = tick_diff / 60.0f;
    }
    
    update_locomotion(dt);
    update_internal_state(dt);
    
    world.set_creature_position(state_.position);
    world.set_creature_orientation(state_.orientation);
    world.apply_motor_force(motor_output_.left_force, motor_output_.right_force);
    
    // Automatically eat any fruit in range
    if (world.consume_if_in_range()) {
        last_satiation_spike_ = 1.0f;
        fruits_eaten_++;
        SPDLOG_INFO("Creature automatically consumed fruit (total: {})", fruits_eaten_);
    }
    
    // Debug: Log creature state periodically
    if (tick % 60 == 0) {
        SPDLOG_DEBUG("Creature tick {}: pos=({:.2f}, {:.2f}), orient={:.2f}, motors=({:.2f}, {:.2f})",
                     tick, state_.position.x, state_.position.y, state_.orientation,
                     motor_output_.left_force, motor_output_.right_force);
    }
    
    last_update_tick_ = tick;
}

void Creature::set_motor_output(const MotorOutput& output) {
    motor_output_ = output;
    motor_output_.left_force = std::max(-1.0f, std::min(1.0f, motor_output_.left_force));
    motor_output_.right_force = std::max(-1.0f, std::min(1.0f, motor_output_.right_force));
}

SensorData Creature::get_sensor_data(const World& world) const {
    SensorData data;
    data.vision_samples = get_vision_samples(world);
    data.forward_velocity = get_proprioception_forward_velocity();
    data.turn_rate = get_proprioception_turn_rate();
    data.hunger_level = get_hunger_level();
    data.last_satiation = last_satiation_spike_;
    
    return data;
}

void Creature::update_locomotion(float dt) {
    float force_diff = motor_output_.right_force - motor_output_.left_force;
    float total_force = (motor_output_.left_force + motor_output_.right_force) * 0.5f;
    
    state_.angular_velocity += force_diff * dt * 10.0f;
    
    float forward_x = std::cos(state_.orientation) * total_force;
    float forward_y = std::sin(state_.orientation) * total_force;
    
    state_.velocity.x += forward_x * dt * 200.0f;
    state_.velocity.y += forward_y * dt * 200.0f;
    
    state_.position = state_.position + state_.velocity * dt;
    state_.orientation += state_.angular_velocity * dt;
    
    state_.velocity = state_.velocity * 0.85f;
    state_.angular_velocity *= 0.80f;
}

void Creature::update_internal_state(float dt) {
    state_.hunger += 0.1f * dt;
    state_.hunger = std::min(1.0f, state_.hunger);
    
    state_.energy -= 0.05f * dt;
    state_.energy = std::max(0.0f, state_.energy);
    
    if (last_satiation_spike_ > 0.0f) {
        last_satiation_spike_ = std::max(0.0f, last_satiation_spike_ - dt * 2.0f);
    }
}

std::vector<VisionSample> Creature::get_vision_samples(const World& world) const {
    return world.get_visible_objects(state_.position, state_.orientation, vision_fov_);
}

float Creature::get_proprioception_forward_velocity() const {
    Vec2 forward_dir(std::cos(state_.orientation), std::sin(state_.orientation));
    return state_.velocity.x * forward_dir.x + state_.velocity.y * forward_dir.y;
}

float Creature::get_proprioception_turn_rate() const {
    return state_.angular_velocity;
}

float Creature::get_hunger_level() const {
    return state_.hunger;
}

} // namespace neuron_creature_experiment