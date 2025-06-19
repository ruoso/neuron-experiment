#include "creature_experiment.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <iostream>
#include <cmath>

namespace neuron_creature_experiment {

CreatureExperiment::CreatureExperiment()
    : window_(nullptr), renderer_(nullptr), running_(false),
      camera_position_(50.0f, 50.0f), simulation_tick_(0),
      show_debug_info_(true), paused_(false),
      left_motor_activation_(0.0f), right_motor_activation_(0.0f) {
    
    initialize_logging();
    last_update_ = std::chrono::steady_clock::now();
}

CreatureExperiment::~CreatureExperiment() {
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
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update_);
    
    if (elapsed.count() >= 16) {
        simulation_tick_++;
        
        // Update motor output based on current activations
        MotorOutput motor_output(left_motor_activation_, right_motor_activation_, false);
        creature_->set_motor_output(motor_output);
        
        creature_->update(simulation_tick_, *world_);
        world_->step_simulation(simulation_tick_);
        
        update_camera();
        
        last_update_ = now;
        
        if (simulation_tick_ % 60 == 0) {
            spdlog::debug("Simulation tick: {}, Creature pos: ({:.2f}, {:.2f})", 
                         simulation_tick_, 
                         creature_->get_position().x, 
                         creature_->get_position().y);
        }
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
    
    if (show_debug_info_) {
        render_debug_info();
    }
    
    SDL_RenderPresent(renderer_);
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

void CreatureExperiment::cleanup() {
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

int main(int argc, char* argv[]) {
    neuron_creature_experiment::CreatureExperiment app;
    app.run();
    return 0;
}