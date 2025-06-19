#include <gtest/gtest.h>
#include "world.h"
#include "creature.h"
#include <cmath>

using namespace neuron_creature_experiment;

class WorldTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.width = 50.0f;
        config_.height = 50.0f;
        config_.simulation_dt = 1.0f / 60.0f;
        config_.max_trees = 10;
        config_.max_fruits = 50;
        config_.fruit_max_satiation = 10.0f;
        config_.creature_eat_radius = 2.0f;
        config_.hunger_increase_rate = 0.1f;
    }
    
    WorldConfig config_;
};

TEST_F(WorldTest, WorldCreation) {
    World world(config_);
    
    EXPECT_EQ(world.get_current_tick(), 0);
    EXPECT_TRUE(world.get_trees().empty());
    EXPECT_TRUE(world.get_fruits().empty());
}

TEST_F(WorldTest, AddTrees) {
    World world(config_);
    
    world.add_tree(Vec2(25.0f, 25.0f));
    EXPECT_EQ(world.get_trees().size(), 1);
    
    const Tree& tree = world.get_trees()[0];
    EXPECT_EQ(tree.state.lifecycle_state, TreeLifecycleState::SEEDLING);
    EXPECT_EQ(tree.position.x, 25.0f);
    EXPECT_EQ(tree.position.y, 25.0f);
}

TEST_F(WorldTest, AddFruits) {
    World world(config_);
    
    world.add_fruit(Vec2(10.0f, 10.0f), 1);
    EXPECT_EQ(world.get_fruits().size(), 1);
    
    const Fruit& fruit = world.get_fruits()[0];
    EXPECT_EQ(fruit.position.x, 10.0f);
    EXPECT_EQ(fruit.position.y, 10.0f);
    EXPECT_EQ(fruit.parent_tree_id, 1);
    EXPECT_TRUE(fruit.available);
    EXPECT_EQ(fruit.maturity, 0.0f);
}

TEST_F(WorldTest, WorldWrapping) {
    World world(config_);
    
    Vec2 pos1(75.0f, 25.0f);
    Vec2 wrapped = world.wrap_position(pos1);
    EXPECT_EQ(wrapped.x, 25.0f);
    EXPECT_EQ(wrapped.y, 25.0f);
    
    Vec2 pos2(-10.0f, 30.0f);
    wrapped = world.wrap_position(pos2);
    EXPECT_EQ(wrapped.x, 40.0f);
    EXPECT_EQ(wrapped.y, 30.0f);
    
    Vec2 pos3(25.0f, -15.0f);
    wrapped = world.wrap_position(pos3);
    EXPECT_EQ(wrapped.x, 25.0f);
    EXPECT_EQ(wrapped.y, 35.0f);
}

TEST_F(WorldTest, DistanceWrapping) {
    World world(config_);
    
    Vec2 pos1(5.0f, 25.0f);
    Vec2 pos2(45.0f, 25.0f);
    
    float distance = world.wrap_distance(pos1, pos2);
    EXPECT_NEAR(distance, 10.0f, 1e-6f);
    
    Vec2 pos3(25.0f, 5.0f);
    Vec2 pos4(25.0f, 45.0f);
    
    distance = world.wrap_distance(pos3, pos4);
    EXPECT_NEAR(distance, 10.0f, 1e-6f);
}

TEST_F(WorldTest, SimulationStep) {
    World world(config_);
    
    world.add_tree(Vec2(25.0f, 25.0f));
    world.step_simulation(1);
    
    EXPECT_EQ(world.get_current_tick(), 1);
    
    const Tree& tree = world.get_trees()[0];
    EXPECT_GT(tree.state.age, 0.0f);
    EXPECT_GT(tree.state.state_timer, 0.0f);
}

class CreatureTest : public ::testing::Test {
protected:
    void SetUp() override {
        start_pos_ = Vec2(10.0f, 10.0f);
        start_orientation_ = 0.0f;
    }
    
    Vec2 start_pos_;
    float start_orientation_;
};

TEST_F(CreatureTest, CreatureCreation) {
    Creature creature(start_pos_, start_orientation_);
    
    EXPECT_EQ(creature.get_position().x, start_pos_.x);
    EXPECT_EQ(creature.get_position().y, start_pos_.y);
    EXPECT_EQ(creature.get_orientation(), start_orientation_);
    EXPECT_EQ(creature.get_hunger(), 0.0f);
    EXPECT_EQ(creature.get_energy(), 1.0f);
}

TEST_F(CreatureTest, MotorOutput) {
    Creature creature(start_pos_, start_orientation_);
    
    MotorOutput output(0.5f, -0.3f, false);
    creature.set_motor_output(output);
    
    MotorOutput retrieved = creature.get_motor_output();
    EXPECT_EQ(retrieved.left_force, 0.5f);
    EXPECT_EQ(retrieved.right_force, -0.3f);
    EXPECT_FALSE(retrieved.eat_action);
}

TEST_F(CreatureTest, MotorOutputClamping) {
    Creature creature(start_pos_, start_orientation_);
    
    MotorOutput output(2.0f, -1.5f, true);
    creature.set_motor_output(output);
    
    MotorOutput retrieved = creature.get_motor_output();
    EXPECT_EQ(retrieved.left_force, 1.0f);
    EXPECT_EQ(retrieved.right_force, -1.0f);
    EXPECT_TRUE(retrieved.eat_action);
}

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.width = 100.0f;
        config_.height = 100.0f;
        config_.creature_eat_radius = 3.0f;
        config_.fruit_max_satiation = 5.0f;
    }
    
    WorldConfig config_;
};

TEST_F(IntegrationTest, CreatureWorldInteraction) {
    World world(config_);
    Creature creature(Vec2(50.0f, 50.0f), 0.0f);
    
    world.add_fruit(Vec2(52.0f, 50.0f), 1);
    
    world.set_creature_position(creature.get_position());
    world.set_creature_orientation(creature.get_orientation());
    
    EXPECT_EQ(world.get_fruits().size(), 1);
    EXPECT_TRUE(world.get_fruits()[0].available);
    
    bool consumed = world.consume_if_in_range();
    EXPECT_TRUE(consumed);
    EXPECT_FALSE(world.get_fruits()[0].available);
}

TEST_F(IntegrationTest, CreatureVisionSystem) {
    World world(config_);
    Creature creature(Vec2(50.0f, 50.0f), 0.0f);
    
    world.add_fruit(Vec2(55.0f, 50.0f), 1);
    world.add_fruit(Vec2(45.0f, 50.0f), 2);
    
    std::vector<VisionSample> samples = world.get_visible_objects(
        creature.get_position(), 
        creature.get_orientation(), 
        M_PI / 3.0f
    );
    
    EXPECT_GT(samples.size(), 0);
    
    for (const auto& sample : samples) {
        EXPECT_GE(sample.total_intensity, 0.0f);
        EXPECT_GE(sample.distance, 0.0f);
        EXPECT_GE(sample.blended_color.r, 0.0f);
        EXPECT_GE(sample.blended_color.g, 0.0f);
        EXPECT_GE(sample.blended_color.b, 0.0f);
    }
}

TEST_F(IntegrationTest, TreeFruitLifecycle) {
    World world(config_);
    
    world.add_tree(Vec2(50.0f, 50.0f));
    EXPECT_EQ(world.get_trees().size(), 1);
    EXPECT_EQ(world.get_fruits().size(), 0);
    
    Tree& tree = const_cast<Tree&>(world.get_trees()[0]);
    tree.state.lifecycle_state = TreeLifecycleState::MATURE;
    tree.state.state_timer = 15.1f;
    
    world.step_simulation(1);
    
    EXPECT_GT(world.get_fruits().size(), 0);
    
    for (const auto& fruit : world.get_fruits()) {
        EXPECT_EQ(fruit.parent_tree_id, tree.tree_id);
        EXPECT_TRUE(fruit.available);
    }
}