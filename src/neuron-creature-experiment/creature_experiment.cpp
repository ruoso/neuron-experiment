#include "creature_experiment.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <iostream>
#include <cmath>
#include <thread>
#include <algorithm>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cctype>
#include <random>

namespace neuron_creature_experiment {

CreatureExperiment::CreatureExperiment(const std::string& layout_encoding)
    : window_(nullptr), renderer_(nullptr), running_(false),
      camera_position_(50.0f, 50.0f), simulation_tick_(0),
      show_debug_info_(true), paused_(false), neural_mode_(true),
      left_motor_activation_(0.0f), right_motor_activation_(0.0f),
      left_motor_feedback_(0.0f), right_motor_feedback_(0.0f),
      left_motor_sent_(false), right_motor_sent_(false), vision_activation_counter_(0),
      left_motor_activators_(0.0f), left_motor_suppressors_(0.0f),
      right_motor_activators_(0.0f), right_motor_suppressors_(0.0f),
      ticks_survived_(0), total_distance_moved_(0.0f), last_position_(50.0f, 50.0f),
      layout_encoding_(layout_encoding) {
    
    initialize_logging();
    
    // Decode sensor/actuator layout
    decode_layout(layout_encoding_);
    
    // Prepare sensor and actuator positions for neural simulation
    std::vector<SensorPosition> sensor_positions;
    std::vector<ActuatorPosition> actuator_positions;
    
    // Add vision sensors with tags
    for (size_t i = 0; i < layout_.vision_sensors.size(); ++i) {
        uint16_t sensor_tag = VISION_SENSOR_TAG_BASE + static_cast<uint16_t>(i);
        sensor_positions.emplace_back(layout_.vision_sensors[i], sensor_tag);
    }
    
    // Add hunger and satiation sensors
    sensor_positions.emplace_back(layout_.hunger_sensor, HUNGER_SENSOR_TAG);
    sensor_positions.emplace_back(layout_.satiation_sensor, SATIATION_SENSOR_TAG);
    
    // Add motor actuators with tags
    for (const auto& activator_pos : layout_.left_motor_activators) {
        actuator_positions.emplace_back(activator_pos, LEFT_MOTOR_ACTIVATOR_TAG);
    }
    for (const auto& suppressor_pos : layout_.left_motor_suppressors) {
        actuator_positions.emplace_back(suppressor_pos, LEFT_MOTOR_SUPPRESSOR_TAG);
    }
    for (const auto& activator_pos : layout_.right_motor_activators) {
        actuator_positions.emplace_back(activator_pos, RIGHT_MOTOR_ACTIVATOR_TAG);
    }
    for (const auto& suppressor_pos : layout_.right_motor_suppressors) {
        actuator_positions.emplace_back(suppressor_pos, RIGHT_MOTOR_SUPPRESSOR_TAG);
    }
    
    // Initialize neural simulation with custom layout
    neural_sim_.initialize(sensor_positions, actuator_positions);
    
    // Set up firing callback for visualization
    neural_sim_.set_firing_callback([this](const std::vector<NeuronFiringEvent>& events) {
        // Neural firing events are automatically handled by the base system
    });
    
    // Start neural simulation threads
    neural_sim_.start();
    
    last_update_ = std::chrono::steady_clock::now();
}

CreatureExperiment::~CreatureExperiment() {
    neural_sim_.stop();
    cleanup();
}

void CreatureExperiment::initialize_logging() {
    try {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::debug);
        console_sink->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
        
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("creature_experiment.log", true);
        file_sink->set_level(spdlog::level::debug);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v");
        
        auto logger = std::make_shared<spdlog::logger>("creature_app", 
                                                      spdlog::sinks_init_list{console_sink, file_sink});
        logger->set_level(spdlog::level::debug);
        
        spdlog::set_default_logger(logger);
        
        spdlog::info("Creature experiment logging system initialized");
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Log initialization failed: " << ex.what() << std::endl;
    }
}

bool CreatureExperiment::initialize() {
    spdlog::info("Initializing Creature Experiment Application...");
    
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        spdlog::error("SDL initialization failed: {}", SDL_GetError());
        return false;
    }
    spdlog::debug("SDL initialized successfully");
    
    window_ = SDL_CreateWindow("Neuron Creature Experiment - 2D World",
                             SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                             WINDOW_WIDTH, WINDOW_HEIGHT,
                             SDL_WINDOW_SHOWN);
    
    if (!window_) {
        spdlog::error("Main window creation failed: {}", SDL_GetError());
        return false;
    }
    spdlog::debug("Main window created: {}x{}", WINDOW_WIDTH, WINDOW_HEIGHT);
    
    renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer_) {
        spdlog::error("Main renderer creation failed: {}", SDL_GetError());
        return false;
    }
    spdlog::debug("Main renderer created successfully");
    
    // Initialize brain visualization
    if (!brain_viz_.initialize()) {
        spdlog::error("Brain visualization initialization failed");
        return false;
    }
    
    initialize_world();
    
    spdlog::info("Application initialization complete");
    return true;
}

void CreatureExperiment::initialize_world() {
    WorldConfig config;
    config.width = 100.0f;
    config.height = 100.0f;
    config.max_trees = 20;
    config.max_fruits = 100;
    config.simulation_dt = 1.0f / 60.0f;
    config.creature_eat_radius = 5.0f;
    
    world_ = std::make_unique<World>(config);
    creature_ = std::make_unique<Creature>(Vec2(50.0f, 50.0f), 0.0f);
    
    for (int i = 0; i < 15; ++i) {
        float x = 10.0f + (i % 5) * 20.0f + (rand() % 10 - 5);
        float y = 10.0f + (i / 5) * 20.0f + (rand() % 10 - 5);
        world_->add_tree(Vec2(x, y));
        
        // Set varied initial age/maturity for trees
        auto& trees = const_cast<std::vector<Tree>&>(world_->get_trees());
        if (!trees.empty()) {
            Tree& tree = trees.back();
            // Randomize age between 0 and 44 seconds (80% of full lifecycle)
            float random_age = (rand() % 4400) / 100.0f; // 0.0 to 44.0 seconds
            tree.state.age = random_age;
            
            // Set appropriate state and state_timer based on age
            if (random_age < 10.0f) {
                tree.state.lifecycle_state = TreeLifecycleState::SEEDLING;
                tree.state.state_timer = random_age;
            } else if (random_age < 25.0f) {
                tree.state.lifecycle_state = TreeLifecycleState::MATURE;
                tree.state.state_timer = random_age - 10.0f;
            } else if (random_age < 45.0f) {
                tree.state.lifecycle_state = TreeLifecycleState::FRUITING;
                tree.state.state_timer = random_age - 25.0f;
            } else {
                tree.state.lifecycle_state = TreeLifecycleState::DORMANT;
                tree.state.state_timer = random_age - 45.0f;
            }
            
            tree.update_color_for_state();
        }
    }
    
    // Fruits will be spawned naturally by trees during their fruiting phase
    
    spdlog::info("World initialized with {} trees and {} fruits", 
                world_->get_trees().size(), world_->get_fruits().size());
}

void CreatureExperiment::run() {
    if (!initialize()) {
        return;
    }
    
    running_ = true;
    
    while (running_) {
        handle_events();
        
        if (!paused_) {
            update();
        }
        
        render();
        SDL_Delay(16);
    }
}

void CreatureExperiment::handle_events() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                running_ = false;
                break;
            
            case SDL_KEYDOWN:
                handle_keypress(event.key.keysym.sym, true);
                break;
                
            case SDL_KEYUP:
                handle_keypress(event.key.keysym.sym, false);
                break;
        }
    }
}

void CreatureExperiment::handle_keypress(SDL_Keycode key, bool pressed) {
    switch (key) {
        case SDLK_ESCAPE:
            if (pressed) running_ = false;
            break;
        
        case SDLK_SPACE:
            if (pressed) {
                paused_ = !paused_;
                spdlog::info("Simulation {}", paused_ ? "paused" : "resumed");
            }
            break;
        
        case SDLK_F1:
            if (pressed) show_debug_info_ = !show_debug_info_;
            break;
        
        case SDLK_w:
            left_motor_activation_ = pressed ? 1.0f : 0.0f;
            break;
            
        case SDLK_d:
            right_motor_activation_ = pressed ? 1.0f : 0.0f;
            break;
        
        
        // E key removed - eating is now automatic when in range
    }
}

void CreatureExperiment::update() {
    // Wait for neural simulation to be ready, then advance
    if (neural_sim_.is_ready_to_advance()) {
        simulation_tick_++;
        
        // Generate sensor activations from creature state
        if (neural_mode_) {
            generate_sensor_activations();
        }
        
        // Process actuator outputs and combine with manual input
        float neural_left = 0.0f, neural_right = 0.0f;
        if (neural_mode_) {
            process_actuator_outputs();
            neural_left = left_motor_feedback_;
            neural_right = right_motor_feedback_;
        }
        
        // Combine neural and manual control (max of both)
        float combined_left = std::max(left_motor_activation_, neural_left);
        float combined_right = std::max(right_motor_activation_, neural_right);
        
        // Update motor output
        MotorOutput motor_output(combined_left, combined_right, false);
        creature_->set_motor_output(motor_output);
        
        creature_->update(simulation_tick_, *world_);
        world_->step_simulation(simulation_tick_);
        
        // Update survival statistics
        ticks_survived_ = simulation_tick_;
        Vec2 current_pos = creature_->get_position();
        total_distance_moved_ += (current_pos - last_position_).magnitude();
        last_position_ = current_pos;
        
        // Check if creature died from hunger
        float hunger = creature_->get_hunger();
        if (hunger >= 10.0f) {
            // Write survival summary and exit
            std::string filename = "survival_summary_" + get_layout_filename_suffix() + ".txt";
            std::ofstream summary_file(filename);
            if (summary_file.is_open()) {
                summary_file << ticks_survived_ << std::endl;
                summary_file << total_distance_moved_ << std::endl;
                summary_file << creature_->get_fruits_eaten() << std::endl;
                summary_file.close();
                spdlog::info("Creature died from hunger after {} ticks. Moved {:.2f} units, ate {} fruits. Summary written to {}", 
                           ticks_survived_, total_distance_moved_, creature_->get_fruits_eaten(), filename);
            }
            running_ = false;
            return;
        }
        
        update_camera();
        
        // Advance neural simulation
        neural_sim_.advance_timestamp();
        
        // Update charts synchronized with simulation timestamps
        brain_viz_.update_charts();
        
        // Fade motor feedback for next cycle
        left_motor_feedback_ *= 0.75f;
        right_motor_feedback_ *= 0.75f;
        
        // Fade visualization values
        left_motor_activators_ *= 0.85f;
        left_motor_suppressors_ *= 0.85f;
        right_motor_activators_ *= 0.85f;
        right_motor_suppressors_ *= 0.85f;
        
        if (left_motor_feedback_ < 0.01f) {
            left_motor_feedback_ = 0.0f;
            left_motor_sent_ = false;
        }
        if (right_motor_feedback_ < 0.01f) {
            right_motor_feedback_ = 0.0f;
            right_motor_sent_ = false;
        }
        
        // Clear visualization values when very small
        if (left_motor_activators_ < 0.01f) left_motor_activators_ = 0.0f;
        if (left_motor_suppressors_ < 0.01f) left_motor_suppressors_ = 0.0f;
        if (right_motor_activators_ < 0.01f) right_motor_activators_ = 0.0f;
        if (right_motor_suppressors_ < 0.01f) right_motor_suppressors_ = 0.0f;
        
        if (simulation_tick_ % 60 == 0) {
            spdlog::debug("Simulation tick: {}, Creature pos: ({:.2f}, {:.2f})", 
                         simulation_tick_, 
                         creature_->get_position().x, 
                         creature_->get_position().y);
        }
    } else {
        // Sleep briefly to avoid busy-waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void CreatureExperiment::update_camera() {
    Vec2 target = creature_->get_position();
    
    camera_position_.x += (target.x - camera_position_.x) * CAMERA_FOLLOW_SPEED;
    camera_position_.y += (target.y - camera_position_.y) * CAMERA_FOLLOW_SPEED;
}

Vec2 CreatureExperiment::world_to_screen(const Vec2& world_pos) const {
    Vec2 relative = world_pos - camera_position_;
    return Vec2(
        WINDOW_WIDTH / 2.0f + relative.x * PIXELS_PER_UNIT,
        WINDOW_HEIGHT / 2.0f + relative.y * PIXELS_PER_UNIT
    );
}

Vec2 CreatureExperiment::screen_to_world(const Vec2& screen_pos) const {
    Vec2 relative(
        (screen_pos.x - WINDOW_WIDTH / 2.0f) / PIXELS_PER_UNIT,
        (screen_pos.y - WINDOW_HEIGHT / 2.0f) / PIXELS_PER_UNIT
    );
    return camera_position_ + relative;
}

void CreatureExperiment::render() {
    render_background();
    render_trees();
    render_fruits();
    render_creature();
    render_creature_vision();
    render_sensor_strips();
    render_neural_overlay();
    
    if (show_debug_info_) {
        render_debug_info();
    }
    
    SDL_RenderPresent(renderer_);
    render_visualization();
}

void CreatureExperiment::render_background() {
    SDL_SetRenderDrawColor(renderer_, 20, 40, 20, 255);
    SDL_RenderClear(renderer_);
    
    SDL_SetRenderDrawColor(renderer_, 30, 50, 30, 255);
    
    for (int x = -10; x <= 10; ++x) {
        Vec2 world_start(camera_position_.x + x * 10.0f, camera_position_.y - 100.0f);
        Vec2 world_end(camera_position_.x + x * 10.0f, camera_position_.y + 100.0f);
        Vec2 screen_start = world_to_screen(world_start);
        Vec2 screen_end = world_to_screen(world_end);
        
        SDL_RenderDrawLine(renderer_, 
                          static_cast<int>(screen_start.x), static_cast<int>(screen_start.y),
                          static_cast<int>(screen_end.x), static_cast<int>(screen_end.y));
    }
    
    for (int y = -10; y <= 10; ++y) {
        Vec2 world_start(camera_position_.x - 100.0f, camera_position_.y + y * 10.0f);
        Vec2 world_end(camera_position_.x + 100.0f, camera_position_.y + y * 10.0f);
        Vec2 screen_start = world_to_screen(world_start);
        Vec2 screen_end = world_to_screen(world_end);
        
        SDL_RenderDrawLine(renderer_, 
                          static_cast<int>(screen_start.x), static_cast<int>(screen_start.y),
                          static_cast<int>(screen_end.x), static_cast<int>(screen_end.y));
    }
}

void CreatureExperiment::render_trees() {
    for (const auto& tree : world_->get_trees()) {
        Vec2 screen_pos = world_to_screen(tree.position);
        
        uint8_t r, g, b;
        get_tree_color(tree, r, g, b);
        
        SDL_SetRenderDrawColor(renderer_, r, g, b, 255);
        
        int radius = static_cast<int>(tree.radius * PIXELS_PER_UNIT);
        
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx * dx + dy * dy <= radius * radius) {
                    SDL_RenderDrawPoint(renderer_, 
                                      static_cast<int>(screen_pos.x) + dx,
                                      static_cast<int>(screen_pos.y) + dy);
                }
            }
        }
    }
}

void CreatureExperiment::render_fruits() {
    for (const auto& fruit : world_->get_fruits()) {
        if (!fruit.available) continue;
        
        Vec2 screen_pos = world_to_screen(fruit.position);
        
        uint8_t r, g, b;
        get_fruit_color(fruit, r, g, b);
        
        SDL_SetRenderDrawColor(renderer_, r, g, b, 255);
        
        int radius = static_cast<int>(fruit.radius * PIXELS_PER_UNIT);
        
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx * dx + dy * dy <= radius * radius) {
                    SDL_RenderDrawPoint(renderer_, 
                                      static_cast<int>(screen_pos.x) + dx,
                                      static_cast<int>(screen_pos.y) + dy);
                }
            }
        }
    }
}

void CreatureExperiment::render_creature() {
    Vec2 screen_pos = world_to_screen(creature_->get_position());
    float orientation = creature_->get_orientation();
    
    SDL_SetRenderDrawColor(renderer_, 100, 150, 255, 255);
    
    int body_radius = static_cast<int>(2.0f * PIXELS_PER_UNIT);
    
    for (int dy = -body_radius; dy <= body_radius; ++dy) {
        for (int dx = -body_radius; dx <= body_radius; ++dx) {
            if (dx * dx + dy * dy <= body_radius * body_radius) {
                SDL_RenderDrawPoint(renderer_, 
                                  static_cast<int>(screen_pos.x) + dx,
                                  static_cast<int>(screen_pos.y) + dy);
            }
        }
    }
    
    SDL_SetRenderDrawColor(renderer_, 255, 255, 100, 255);
    
    float nose_length = 3.0f * PIXELS_PER_UNIT;
    Vec2 nose_end(
        screen_pos.x + std::cos(orientation) * nose_length,
        screen_pos.y + std::sin(orientation) * nose_length
    );
    
    SDL_RenderDrawLine(renderer_, 
                      static_cast<int>(screen_pos.x), static_cast<int>(screen_pos.y),
                      static_cast<int>(nose_end.x), static_cast<int>(nose_end.y));
}

void CreatureExperiment::render_creature_vision() {
    if (!show_debug_info_) return;
    
    Vec2 screen_pos = world_to_screen(creature_->get_position());
    float orientation = creature_->get_orientation();
    float fov = M_PI / 3.0f;
    float vision_range = 20.0f * PIXELS_PER_UNIT;
    
    SDL_SetRenderDrawColor(renderer_, 255, 255, 0, 50);
    
    float start_angle = orientation - fov * 0.5f;
    float end_angle = orientation + fov * 0.5f;
    
    for (float angle = start_angle; angle <= end_angle; angle += 0.1f) {
        Vec2 ray_end(
            screen_pos.x + std::cos(angle) * vision_range,
            screen_pos.y + std::sin(angle) * vision_range
        );
        
        SDL_RenderDrawLine(renderer_, 
                          static_cast<int>(screen_pos.x), static_cast<int>(screen_pos.y),
                          static_cast<int>(ray_end.x), static_cast<int>(ray_end.y));
    }
}

void CreatureExperiment::render_sensor_strips() {
    const int STRIP_WIDTH = WINDOW_WIDTH - 40;
    const int STRIP_HEIGHT = 20;
    const int STRIP_Y_START = 20;
    const int STRIP_SPACING = 30;
    const int NUM_VISION_STRIPS = neuron_creature_experiment::NUM_VISION_STRIPS;
    
    // Get vision data from creature - this should be synchronized with simulation
    SensorData sensor_data = creature_->get_sensor_data(*world_);
    
    // Debug: Check if we have vision data
    static uint32_t render_count = 0;
    render_count++;
    if (render_count % 60 == 0) {
        int active_samples = 0;
        for (const auto& sample : sensor_data.vision_samples) {
            if (sample.total_intensity > 0.0f) active_samples++;
        }
        spdlog::debug("Render frame {}: {} active vision samples out of {}", 
                     render_count, active_samples, sensor_data.vision_samples.size());
    }
    
    // Calculate strip segment width
    int segment_width = STRIP_WIDTH / NUM_VISION_STRIPS;
    
    // Draw strip backgrounds
    SDL_SetRenderDrawColor(renderer_, 40, 40, 40, 255);
    for (int strip = 0; strip < 3; ++strip) {
        SDL_Rect bg_rect = {
            20, 
            STRIP_Y_START + strip * STRIP_SPACING, 
            STRIP_WIDTH, 
            STRIP_HEIGHT
        };
        SDL_RenderFillRect(renderer_, &bg_rect);
    }
    
    // Draw vision data for each strip
    for (int i = 0; i < NUM_VISION_STRIPS && i < static_cast<int>(sensor_data.vision_samples.size()); ++i) {
        const VisionSample& sample = sensor_data.vision_samples[i];
        
        // Calculate segment position
        int x = 20 + i * segment_width;
        
        // Extract RGB intensities from blended color
        float red_intensity = sample.blended_color.r;
        float green_intensity = sample.blended_color.g;
        float blue_intensity = sample.blended_color.b;
        
        // Draw red strip
        if (red_intensity > 0.0f) {
            uint8_t intensity = static_cast<uint8_t>(std::min(255.0f, red_intensity * 255.0f));
            int segment_height = static_cast<int>(std::min(1.0f, red_intensity) * STRIP_HEIGHT);
            SDL_SetRenderDrawColor(renderer_, intensity, 0, 0, 255);
            SDL_Rect red_rect = {
                x, 
                STRIP_Y_START + STRIP_HEIGHT - segment_height, 
                segment_width - 2, 
                segment_height
            };
            SDL_RenderFillRect(renderer_, &red_rect);
        }
        
        // Draw green strip
        if (green_intensity > 0.0f) {
            uint8_t intensity = static_cast<uint8_t>(std::min(255.0f, green_intensity * 255.0f));
            int segment_height = static_cast<int>(std::min(1.0f, green_intensity) * STRIP_HEIGHT);
            SDL_SetRenderDrawColor(renderer_, 0, intensity, 0, 255);
            SDL_Rect green_rect = {
                x, 
                STRIP_Y_START + STRIP_SPACING + STRIP_HEIGHT - segment_height, 
                segment_width - 2, 
                segment_height
            };
            SDL_RenderFillRect(renderer_, &green_rect);
        }
        
        // Draw blue strip
        if (blue_intensity > 0.0f) {
            uint8_t intensity = static_cast<uint8_t>(std::min(255.0f, blue_intensity * 255.0f));
            int segment_height = static_cast<int>(std::min(1.0f, blue_intensity) * STRIP_HEIGHT);
            SDL_SetRenderDrawColor(renderer_, 0, 0, intensity, 255);
            SDL_Rect blue_rect = {
                x, 
                STRIP_Y_START + 2 * STRIP_SPACING + STRIP_HEIGHT - segment_height, 
                segment_width - 2, 
                segment_height
            };
            SDL_RenderFillRect(renderer_, &blue_rect);
        }
    }
    
    // Draw strip labels
    SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 255);
    
    // Draw grid lines for segments
    for (int i = 0; i <= NUM_VISION_STRIPS; ++i) {
        int x = 20 + i * segment_width;
        for (int strip = 0; strip < 3; ++strip) {
            int y_start = STRIP_Y_START + strip * STRIP_SPACING;
            SDL_RenderDrawLine(renderer_, x, y_start, x, y_start + STRIP_HEIGHT);
        }
    }
}

void CreatureExperiment::render_debug_info() {
    
}

void CreatureExperiment::get_tree_color(const Tree& tree, uint8_t& r, uint8_t& g, uint8_t& b) const {
    // Use the linear color that was calculated in update_color_for_state()
    r = static_cast<uint8_t>(tree.color.r * 255.0f);
    g = static_cast<uint8_t>(tree.color.g * 255.0f);
    b = static_cast<uint8_t>(tree.color.b * 255.0f);
}

void CreatureExperiment::get_fruit_color(const Fruit& fruit, uint8_t& r, uint8_t& g, uint8_t& b) const {
    // Use the linear color that was calculated in update_color_for_maturity()
    r = static_cast<uint8_t>(fruit.color.r * 255.0f);
    g = static_cast<uint8_t>(fruit.color.g * 255.0f);
    b = static_cast<uint8_t>(fruit.color.b * 255.0f);
}

void CreatureExperiment::generate_sensor_activations() {
    std::vector<SensorActivation> activations;
    uint32_t current_timestamp = neural_sim_.get_current_timestamp();
    
    // Get sensor data from creature
    SensorData sensor_data = creature_->get_sensor_data(*world_);
    
    // 1. Vision sensor activations (192 sensors using dynamic layout) - only every 10 ticks
    vision_activation_counter_++;
    if (vision_activation_counter_ >= 10) {
        vision_activation_counter_ = 0;
        
        const auto& vision_samples = sensor_data.vision_samples;
        for (int strip = 0; strip < static_cast<int>(vision_samples.size()) && strip < 64; ++strip) {
            const auto& sample = vision_samples[strip];
            
            // Generate R, G, B activations for this strip
            float intensities[3] = {sample.blended_color.r, sample.blended_color.g, sample.blended_color.b};
            
            for (int color = 0; color < 3; ++color) {
                if (intensities[color] > 0.01f) {
                    // Map strip + color to sensor tag
                    int sensor_idx = strip * 3 + color;
                    if (sensor_idx < NUM_VISION_SENSORS) {
                        uint16_t sensor_tag = VISION_SENSOR_TAG_BASE + static_cast<uint16_t>(sensor_idx);
                        
                        uint8_t mode_bitmap = 0;
                        float intensity = intensities[color];
                        if (intensity > 0.75f) mode_bitmap |= (1 << 0);        // Mode 0: Very bright
                        else if (intensity > 0.5f) mode_bitmap |= (1 << 1);   // Mode 1: Bright
                        else if (intensity > 0.25f) mode_bitmap |= (1 << 2);  // Mode 2: Medium
                        else mode_bitmap |= (1 << 3);                         // Mode 3: Dim
                        
                        activations.emplace_back(sensor_tag, mode_bitmap, intensity);
                    }
                }
            }
        }
    }
    
    // 2. Hunger sensor activation - only every 10 ticks (same as vision)
    if (vision_activation_counter_ == 0) {
        float hunger = sensor_data.hunger_level;
        if (hunger > 0.01f) {
            uint8_t mode_bitmap = 0;
            
            if (hunger > 0.75f) mode_bitmap |= (1 << 0);
            else if (hunger > 0.5f) mode_bitmap |= (1 << 1);
            else if (hunger > 0.25f) mode_bitmap |= (1 << 2);
            else mode_bitmap |= (1 << 3);
            
            activations.emplace_back(HUNGER_SENSOR_TAG, mode_bitmap, hunger);
        }
        
        // 3. Satiation sensor activation - only every 10 ticks (same as vision)
        float satiation = sensor_data.last_satiation;
        if (satiation > 0.01f) {
            uint8_t mode_bitmap = 0;
            
            if (satiation > 0.75f) mode_bitmap |= (1 << 0);
            else if (satiation > 0.5f) mode_bitmap |= (1 << 1);
            else if (satiation > 0.25f) mode_bitmap |= (1 << 2);
            else mode_bitmap |= (1 << 3);
            
            activations.emplace_back(SATIATION_SENSOR_TAG, mode_bitmap, satiation);
        }
    }
    
    // Process sensor activations and send to neural network
    auto targeted_activations = process_sensor_activations(neural_sim_.get_brain().sensor_grid, activations, current_timestamp);
    
    // Track sensor activation count for chart
    brain_viz_.track_sensor_activations(static_cast<int>(activations.size()));
    
    if (!targeted_activations.empty()) {
        spdlog::debug("Generated {} sensor activations -> {} targeted activations", 
                     activations.size(), targeted_activations.size());
        neural_sim_.send_sensor_activations(targeted_activations);
    }
}

void CreatureExperiment::process_actuator_outputs() {
    // Get all actuation events from neural network
    auto actuation_events = neural_sim_.get_actuator_events();
    
    if (!actuation_events.empty()) {
        spdlog::debug("Processing {} actuator outputs", actuation_events.size());
    }
    
    float left_motor_activators = 0.0f;
    float left_motor_suppressors = 0.0f;
    float right_motor_activators = 0.0f;
    float right_motor_suppressors = 0.0f;
    
    for (const auto& event : actuation_events) {
        // Use the actuator tag to determine motor action
        switch (event.actuator_tag) {
            case LEFT_MOTOR_ACTIVATOR_TAG:
                left_motor_activators += 1.0f;
                spdlog::debug("Left motor ACTIVATED by actuator at ({:.3f}, {:.3f}, {:.3f})", 
                             event.position.x, event.position.y, event.position.z);
                break;
                
            case LEFT_MOTOR_SUPPRESSOR_TAG:
                left_motor_suppressors += 1.0f;
                spdlog::debug("Left motor SUPPRESSED by actuator at ({:.3f}, {:.3f}, {:.3f})", 
                             event.position.x, event.position.y, event.position.z);
                break;
                
            case RIGHT_MOTOR_ACTIVATOR_TAG:
                right_motor_activators += 1.0f;
                spdlog::debug("Right motor ACTIVATED by actuator at ({:.3f}, {:.3f}, {:.3f})", 
                             event.position.x, event.position.y, event.position.z);
                break;
                
            case RIGHT_MOTOR_SUPPRESSOR_TAG:
                right_motor_suppressors += 1.0f;
                spdlog::debug("Right motor SUPPRESSED by actuator at ({:.3f}, {:.3f}, {:.3f})", 
                             event.position.x, event.position.y, event.position.z);
                break;
                
            default:
                // Unknown actuator tag, ignore
                spdlog::debug("Unknown actuator tag {} at ({:.3f}, {:.3f}, {:.3f})", 
                             event.actuator_tag, event.position.x, event.position.y, event.position.z);
                break;
        }
    }
    
    // Calculate net motor activation: activators - suppressors, clamped to [0, 1]
    float left_net_activation = std::max(0.0f, std::min(1.0f, left_motor_activators - left_motor_suppressors));
    float right_net_activation = std::max(0.0f, std::min(1.0f, right_motor_activators - right_motor_suppressors));
    
    // Store neural motor activations for combining with manual input
    left_motor_feedback_ = left_net_activation;
    right_motor_feedback_ = right_net_activation;
    
    // Store for visualization (with decay)
    left_motor_activators_ = left_motor_activators;
    left_motor_suppressors_ = left_motor_suppressors;
    right_motor_activators_ = right_motor_activators;
    right_motor_suppressors_ = right_motor_suppressors;
    
    if (left_motor_activators > 0 || left_motor_suppressors > 0 || right_motor_activators > 0 || right_motor_suppressors > 0) {
        spdlog::debug("Motor summary: Left({:.1f}a - {:.1f}s = {:.2f}), Right({:.1f}a - {:.1f}s = {:.2f})",
                     left_motor_activators, left_motor_suppressors, left_net_activation,
                     right_motor_activators, right_motor_suppressors, right_net_activation);
    }
}


void CreatureExperiment::render_neural_overlay() {
    // Get sensor data from creature
    SensorData sensor_data = creature_->get_sensor_data(*world_);
    
    // Show hunger and satiation levels
    float hunger = sensor_data.hunger_level;
    float satiation = sensor_data.last_satiation;
    
    // Hunger bar (red)
    SDL_SetRenderDrawColor(renderer_, 255, 0, 0, 255);
    SDL_Rect hunger_rect = {10, 750, static_cast<int>(200 * hunger), 20};
    SDL_RenderFillRect(renderer_, &hunger_rect);
    
    // Satiation bar (green)
    SDL_SetRenderDrawColor(renderer_, 0, 255, 0, 255);
    SDL_Rect satiation_rect = {10, 775, static_cast<int>(200 * satiation), 20};
    SDL_RenderFillRect(renderer_, &satiation_rect);
    
    // Left motor visualization - Activators (green) and Suppressors (red)
    int left_base_x = 250;
    int left_base_y = 750;
    
    // Left motor activators (green)
    if (left_motor_activators_ > 0.01f) {
        uint8_t intensity = static_cast<uint8_t>(std::min(255.0f, left_motor_activators_ * 64.0f)); // Scale for visibility
        SDL_SetRenderDrawColor(renderer_, 0, intensity, 0, 255);
        SDL_Rect left_act_rect = {left_base_x, left_base_y, 25, 20};
        SDL_RenderFillRect(renderer_, &left_act_rect);
    }
    
    // Left motor suppressors (red)
    if (left_motor_suppressors_ > 0.01f) {
        uint8_t intensity = static_cast<uint8_t>(std::min(255.0f, left_motor_suppressors_ * 64.0f)); // Scale for visibility
        SDL_SetRenderDrawColor(renderer_, intensity, 0, 0, 255);
        SDL_Rect left_sup_rect = {left_base_x + 25, left_base_y, 25, 20};
        SDL_RenderFillRect(renderer_, &left_sup_rect);
    }
    
    // Left motor net result (cyan - shows final combined effect)
    if (left_motor_feedback_ > 0.01f) {
        uint8_t intensity = static_cast<uint8_t>(left_motor_feedback_ * 255);
        SDL_SetRenderDrawColor(renderer_, 0, intensity, intensity, 255);
        SDL_Rect left_net_rect = {left_base_x, left_base_y + 25, 50, 10};
        SDL_RenderFillRect(renderer_, &left_net_rect);
    }
    
    // Right motor visualization - Activators (green) and Suppressors (red)
    int right_base_x = 320;
    int right_base_y = 750;
    
    // Right motor activators (green)
    if (right_motor_activators_ > 0.01f) {
        uint8_t intensity = static_cast<uint8_t>(std::min(255.0f, right_motor_activators_ * 64.0f)); // Scale for visibility
        SDL_SetRenderDrawColor(renderer_, 0, intensity, 0, 255);
        SDL_Rect right_act_rect = {right_base_x, right_base_y, 25, 20};
        SDL_RenderFillRect(renderer_, &right_act_rect);
    }
    
    // Right motor suppressors (red)
    if (right_motor_suppressors_ > 0.01f) {
        uint8_t intensity = static_cast<uint8_t>(std::min(255.0f, right_motor_suppressors_ * 64.0f)); // Scale for visibility
        SDL_SetRenderDrawColor(renderer_, intensity, 0, 0, 255);
        SDL_Rect right_sup_rect = {right_base_x + 25, right_base_y, 25, 20};
        SDL_RenderFillRect(renderer_, &right_sup_rect);
    }
    
    // Right motor net result (cyan - shows final combined effect)
    if (right_motor_feedback_ > 0.01f) {
        uint8_t intensity = static_cast<uint8_t>(right_motor_feedback_ * 255);
        SDL_SetRenderDrawColor(renderer_, 0, intensity, intensity, 255);
        SDL_Rect right_net_rect = {right_base_x, right_base_y + 25, 50, 10};
        SDL_RenderFillRect(renderer_, &right_net_rect);
    }
    
    // Neural mode indicator
    SDL_SetRenderDrawColor(renderer_, neural_mode_ ? 0 : 128, neural_mode_ ? 255 : 128, 0, 255);
    SDL_Rect neural_rect = {400, 750, 100, 45};
    SDL_RenderFillRect(renderer_, &neural_rect);
}

void CreatureExperiment::render_visualization() {
    brain_viz_.render(neural_sim_);
}

void CreatureExperiment::decode_layout(const std::string& base64_encoding) {
    // Decode base64 to bytes
    std::vector<uint8_t> decoded_bytes;
    
    // Simple base64 decode table
    const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<int> decode_table(256, -1);
    for (int i = 0; i < 64; i++) {
        decode_table[base64_chars[i]] = i;
    }
    
    // Decode base64 string to bytes
    for (size_t i = 0; i < base64_encoding.length(); i += 4) {
        uint32_t packed = 0;
        int padding = 0;
        
        for (int j = 0; j < 4; j++) {
            if (i + j < base64_encoding.length() && base64_encoding[i + j] != '=') {
                packed = (packed << 6) | decode_table[base64_encoding[i + j]];
            } else {
                packed <<= 6;
                padding++;
            }
        }
        
        for (int j = 2; j >= padding; j--) {
            decoded_bytes.push_back((packed >> (j * 8)) & 0xFF);
        }
    }
    
    // Convert bytes to unsigned shorts (little endian)
    std::vector<uint16_t> values;
    for (size_t i = 0; i + 1 < decoded_bytes.size(); i += 2) {
        uint16_t value = decoded_bytes[i] | (decoded_bytes[i + 1] << 8);
        values.push_back(value);
    }
    
    // Calculate expected number of values: 3 coordinates per position
    size_t expected_positions = NUM_VISION_SENSORS + (NUM_MOTOR_ACTUATORS/2) * 4 + 2; // vision + 4 motor arrays + hunger + satiation
    size_t expected_values = expected_positions * 3; // 3 dimensions per position
    
    if (values.size() < expected_values) {
        spdlog::error("Insufficient layout data: got {} values, expected {}", values.size(), expected_values);
        throw std::runtime_error("Invalid layout encoding");
    }
    
    // Map unsigned shorts to brain space coordinates (-1.0 to 1.0)
    auto map_to_brain_space = [](uint16_t value) -> float {
        return (static_cast<float>(value) / 65535.0f) * 2.0f - 1.0f;
    };
    
    // Decode all positions
    size_t value_idx = 0;
    
    // Vision sensors (192 positions)
    for (size_t i = 0; i < NUM_VISION_SENSORS; i++) {
        layout_.vision_sensors[i] = Vec3(
            map_to_brain_space(values[value_idx++]),
            map_to_brain_space(values[value_idx++]),
            map_to_brain_space(values[value_idx++])
        );
    }
    
    // Left motor activators (8 positions)
    for (size_t i = 0; i < NUM_MOTOR_ACTUATORS/2; i++) {
        layout_.left_motor_activators[i] = Vec3(
            map_to_brain_space(values[value_idx++]),
            map_to_brain_space(values[value_idx++]),
            map_to_brain_space(values[value_idx++])
        );
    }
    
    // Left motor suppressors (8 positions)
    for (size_t i = 0; i < NUM_MOTOR_ACTUATORS/2; i++) {
        layout_.left_motor_suppressors[i] = Vec3(
            map_to_brain_space(values[value_idx++]),
            map_to_brain_space(values[value_idx++]),
            map_to_brain_space(values[value_idx++])
        );
    }
    
    // Right motor activators (8 positions)
    for (size_t i = 0; i < NUM_MOTOR_ACTUATORS/2; i++) {
        layout_.right_motor_activators[i] = Vec3(
            map_to_brain_space(values[value_idx++]),
            map_to_brain_space(values[value_idx++]),
            map_to_brain_space(values[value_idx++])
        );
    }
    
    // Right motor suppressors (8 positions)
    for (size_t i = 0; i < NUM_MOTOR_ACTUATORS/2; i++) {
        layout_.right_motor_suppressors[i] = Vec3(
            map_to_brain_space(values[value_idx++]),
            map_to_brain_space(values[value_idx++]),
            map_to_brain_space(values[value_idx++])
        );
    }
    
    // Hunger sensor (1 position)
    layout_.hunger_sensor = Vec3(
        map_to_brain_space(values[value_idx++]),
        map_to_brain_space(values[value_idx++]),
        map_to_brain_space(values[value_idx++])
    );
    
    // Satiation sensor (1 position)
    layout_.satiation_sensor = Vec3(
        map_to_brain_space(values[value_idx++]),
        map_to_brain_space(values[value_idx++]),
        map_to_brain_space(values[value_idx++])
    );
    
    spdlog::info("Successfully decoded layout from base64 encoding ({} values -> {} positions)",
                 values.size(), expected_positions);
}

std::string CreatureExperiment::get_layout_filename_suffix() const {
    // Use first 8 characters of base64 encoding as suffix
    std::string suffix = layout_encoding_.substr(0, 8);
    // Replace any non-alphanumeric characters with underscores
    for (char& c : suffix) {
        if (!std::isalnum(c)) {
            c = '_';
        }
    }
    return suffix;
}

void CreatureExperiment::cleanup() {
    brain_viz_.cleanup();
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

} // namespace neuron_creature_experiment

std::string generate_random_layout_encoding() {
    using namespace neuron_creature_experiment;
    // Calculate total number of unsigned shorts needed
    constexpr int total_positions = NUM_VISION_SENSORS + 8 * 4 + 2; // vision + 4 motor types (8 each) + hunger + satiation
    constexpr int values_per_position = 3; // x, y, z
    constexpr int total_values = total_positions * values_per_position;
    
    // Generate random unsigned short values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 65535);
    
    std::vector<uint16_t> values;
    values.reserve(total_values);
    for (int i = 0; i < total_values; ++i) {
        values.push_back(dis(gen));
    }
    
    // Convert to base64
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(values.data());
    size_t byte_count = values.size() * sizeof(uint16_t);
    
    std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    
    for (size_t i = 0; i < byte_count; i += 3) {
        uint32_t group = 0;
        int group_size = 0;
        
        for (int j = 0; j < 3 && (i + j) < byte_count; ++j) {
            group = (group << 8) | bytes[i + j];
            group_size++;
        }
        
        // Pad to 24 bits
        group <<= (3 - group_size) * 8;
        
        // Extract 6-bit chunks
        for (int j = 3; j >= 0; --j) {
            if (j < 4 - ((3 - group_size) * 4 / 3)) {
                result += base64_chars[(group >> (j * 6)) & 0x3F];
            } else {
                result += '=';
            }
        }
    }
    
    return result;
}

int main(int argc, char* argv[]) {
    std::string layout_encoding;
    
    if (argc == 1) {
        // Generate random layout if no argument provided
        layout_encoding = generate_random_layout_encoding();
        std::cout << "Using random layout: " << layout_encoding << std::endl;
    } else if (argc == 2) {
        layout_encoding = argv[1];
    } else {
        std::cerr << "Usage: " << argv[0] << " [base64_layout_encoding]" << std::endl;
        std::cerr << "If no encoding is provided, a random layout will be generated." << std::endl;
        return 1;
    }
    
    neuron_creature_experiment::CreatureExperiment app(layout_encoding);
    app.run();
    return 0;
}