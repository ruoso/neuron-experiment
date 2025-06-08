#include <gtest/gtest.h>
#include "flow_field.h"
#include <cmath>

using namespace neuronlib;

class FlowFieldTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test flow field: 10x10x10 space
        // Input section: 0-2 (width 2)
        // Output section: 8-10 (width 2) 
        // Middle section: 2-8 (width 6)
        test_field = FlowField3D(0.0f, 0.0f, 0.0f,    // min
                                10.0f, 10.0f, 10.0f,   // max
                                2.0f, 2.0f,            // input/output widths
                                0.1f);                 // resolution
    }
    
    FlowField3D test_field;
    
    float vector_magnitude(const Vec3& v) {
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }
    
    bool vectors_approximately_equal(const Vec3& a, const Vec3& b, float tolerance = 1e-6f) {
        return std::abs(a.x - b.x) < tolerance &&
               std::abs(a.y - b.y) < tolerance &&
               std::abs(a.z - b.z) < tolerance;
    }
};

TEST_F(FlowFieldTest, InputSection_PurePositiveZ) {
    // Test positions in input section (z <= 2)
    Vec3 pos1{5.0f, 5.0f, 0.0f};  // at min_z
    Vec3 pos2{3.0f, 7.0f, 1.0f};  // middle of input section
    Vec3 pos3{8.0f, 2.0f, 2.0f};  // at input section boundary
    
    Vec3 expected{0.0f, 0.0f, 1.0f};
    
    EXPECT_TRUE(vectors_approximately_equal(evaluate_flow_field(test_field, pos1), expected));
    EXPECT_TRUE(vectors_approximately_equal(evaluate_flow_field(test_field, pos2), expected));
    EXPECT_TRUE(vectors_approximately_equal(evaluate_flow_field(test_field, pos3), expected));
}

TEST_F(FlowFieldTest, OutputSection_PurePositiveZ) {
    // Test positions in output section (z >= 8)
    Vec3 pos1{5.0f, 5.0f, 8.0f};   // at output section boundary
    Vec3 pos2{3.0f, 7.0f, 9.0f};   // middle of output section
    Vec3 pos3{8.0f, 2.0f, 10.0f};  // at max_z
    
    Vec3 expected{0.0f, 0.0f, 1.0f};
    
    EXPECT_TRUE(vectors_approximately_equal(evaluate_flow_field(test_field, pos1), expected));
    EXPECT_TRUE(vectors_approximately_equal(evaluate_flow_field(test_field, pos2), expected));
    EXPECT_TRUE(vectors_approximately_equal(evaluate_flow_field(test_field, pos3), expected));
}

TEST_F(FlowFieldTest, MiddleSection_CircularFlow) {
    // Test positions in middle section (2 < z < 8)
    Vec3 center{5.0f, 5.0f, 5.0f};  // center of field
    
    // Test point to the right of center (positive x)
    Vec3 pos_right{7.0f, 5.0f, 5.0f};
    Vec3 flow_right = evaluate_flow_field(test_field, pos_right);
    
    // At positive x, clockwise flow should point in negative y direction
    EXPECT_LT(flow_right.y, 0.0f);
    EXPECT_NEAR(flow_right.x, 0.0f, 1e-6f);
    EXPECT_EQ(flow_right.z, 0.1f);
    
    // Test point above center (positive y)
    Vec3 pos_above{5.0f, 7.0f, 5.0f};
    Vec3 flow_above = evaluate_flow_field(test_field, pos_above);
    
    // At positive y, clockwise flow should point in positive x direction
    EXPECT_GT(flow_above.x, 0.0f);
    EXPECT_NEAR(flow_above.y, 0.0f, 1e-6f);
    EXPECT_EQ(flow_above.z, 0.1f);
    
    // Test point to the left of center (negative x)
    Vec3 pos_left{3.0f, 5.0f, 5.0f};
    Vec3 flow_left = evaluate_flow_field(test_field, pos_left);
    
    // At negative x, clockwise flow should point in positive y direction
    EXPECT_GT(flow_left.y, 0.0f);
    EXPECT_NEAR(flow_left.x, 0.0f, 1e-6f);
    EXPECT_EQ(flow_left.z, 0.1f);
    
    // Test point below center (negative y)
    Vec3 pos_below{5.0f, 3.0f, 5.0f};
    Vec3 flow_below = evaluate_flow_field(test_field, pos_below);
    
    // At negative y, clockwise flow should point in negative x direction
    EXPECT_LT(flow_below.x, 0.0f);
    EXPECT_NEAR(flow_below.y, 0.0f, 1e-6f);
    EXPECT_EQ(flow_below.z, 0.1f);
}

TEST_F(FlowFieldTest, MiddleSection_FlowVectorNormalized) {
    // Test that flow vectors in middle section are normalized (except z component)
    Vec3 pos{7.0f, 8.0f, 5.0f};  // arbitrary position in middle section
    Vec3 flow = evaluate_flow_field(test_field, pos);
    
    // Calculate magnitude of x,y components
    float xy_magnitude = std::sqrt(flow.x * flow.x + flow.y * flow.y);
    EXPECT_NEAR(xy_magnitude, 1.0f, 1e-6f);
    
    // Z component should be fixed at 0.1
    EXPECT_EQ(flow.z, 0.1f);
}

TEST_F(FlowFieldTest, MiddleSection_AtCenter) {
    // Test behavior at exact center (should handle zero magnitude gracefully)
    Vec3 center{5.0f, 5.0f, 5.0f};
    Vec3 flow = evaluate_flow_field(test_field, center);
    
    // At center, relative position is (0,0), so flow should be (0,0,0.1)
    EXPECT_EQ(flow.x, 0.0f);
    EXPECT_EQ(flow.y, 0.0f);
    EXPECT_EQ(flow.z, 0.1f);
}

TEST_F(FlowFieldTest, SectionBoundaries) {
    // Test exact boundary conditions
    
    // Just inside input section
    Vec3 pos_input{5.0f, 5.0f, 1.99f};
    Vec3 flow_input = evaluate_flow_field(test_field, pos_input);
    EXPECT_TRUE(vectors_approximately_equal(flow_input, {0.0f, 0.0f, 1.0f}));
    
    // Just inside middle section from input side
    Vec3 pos_middle_start{5.0f, 5.0f, 2.01f};
    Vec3 flow_middle_start = evaluate_flow_field(test_field, pos_middle_start);
    EXPECT_EQ(flow_middle_start.z, 0.1f);  // Should be in circular flow
    
    // Just inside middle section from output side
    Vec3 pos_middle_end{5.0f, 5.0f, 7.99f};
    Vec3 flow_middle_end = evaluate_flow_field(test_field, pos_middle_end);
    EXPECT_EQ(flow_middle_end.z, 0.1f);  // Should be in circular flow
    
    // Just inside output section
    Vec3 pos_output{5.0f, 5.0f, 8.01f};
    Vec3 flow_output = evaluate_flow_field(test_field, pos_output);
    EXPECT_TRUE(vectors_approximately_equal(flow_output, {0.0f, 0.0f, 1.0f}));
}

TEST_F(FlowFieldTest, FlowField_ConstructorDefaults) {
    FlowField3D default_field;
    
    EXPECT_EQ(default_field.min_x, 0.0f);
    EXPECT_EQ(default_field.min_y, 0.0f);
    EXPECT_EQ(default_field.min_z, 0.0f);
    EXPECT_EQ(default_field.max_x, 1.0f);
    EXPECT_EQ(default_field.max_y, 1.0f);
    EXPECT_EQ(default_field.max_z, 1.0f);
    EXPECT_EQ(default_field.input_section_width, 0.2f);
    EXPECT_EQ(default_field.output_section_width, 0.2f);
    EXPECT_EQ(default_field.resolution, 0.1f);
}

TEST_F(FlowFieldTest, CircularFlow_Orthogonality) {
    // Test that circular flow is orthogonal to radius vector
    Vec3 center{5.0f, 5.0f, 5.0f};
    Vec3 pos{7.0f, 8.0f, 5.0f};  // arbitrary position in middle section
    
    Vec3 flow = evaluate_flow_field(test_field, pos);
    Vec3 radius{pos.x - center.x, pos.y - center.y, 0.0f};  // radius vector
    
    // Flow (x,y components) should be orthogonal to radius
    float dot_product = flow.x * radius.x + flow.y * radius.y;
    EXPECT_NEAR(dot_product, 0.0f, 1e-6f);
}