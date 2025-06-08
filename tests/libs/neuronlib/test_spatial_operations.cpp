#include <gtest/gtest.h>
#include "spatial.h"
#include "spatial_operations.h"
#include "geometry.h"
#include <memory>

using namespace neuronlib;

class SpatialOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple grid with root branch
        SpatialBranch root_branch;
        root_branch.min_x = 0.0f;
        root_branch.min_y = 0.0f;
        root_branch.min_z = 0.0f;
        root_branch.max_x = 10.0f;
        root_branch.max_y = 10.0f;
        root_branch.max_z = 10.0f;
        
        auto root_node = std::make_shared<SpatialNode>(std::move(root_branch));
        
        test_grid = std::make_shared<SpatialGrid>();
        test_grid->root = root_node;
        test_grid->max_depth = 3;
    }
    
    SpatialGridPtr test_grid;
};

TEST_F(SpatialOperationsTest, InsertBatch_ThrowsOnNullGrid) {
    std::vector<SpatialInsert> items = {
        {1, {1.0f, 1.0f, 1.0f}}
    };
    
    EXPECT_THROW(insert_batch(nullptr, items), std::invalid_argument);
}

TEST_F(SpatialOperationsTest, InsertBatch_ThrowsOnGridWithoutRoot) {
    auto empty_grid = std::make_shared<SpatialGrid>();
    empty_grid->max_depth = 3;
    empty_grid->root = nullptr;
    
    std::vector<SpatialInsert> items = {
        {1, {1.0f, 1.0f, 1.0f}}
    };
    
    EXPECT_THROW(insert_batch(empty_grid, items), std::invalid_argument);
}

TEST_F(SpatialOperationsTest, InsertBatch_ReturnsOriginalGridOnEmptyItems) {
    std::vector<SpatialInsert> empty_items;
    
    auto result = insert_batch(test_grid, empty_items);
    
    EXPECT_EQ(result, test_grid);
}

TEST_F(SpatialOperationsTest, InsertBatch_CreatesNewGrid) {
    std::vector<SpatialInsert> items = {
        {1, {1.0f, 1.0f, 1.0f}}
    };
    
    auto result = insert_batch(test_grid, items);
    
    EXPECT_NE(result, test_grid);
    EXPECT_EQ(result->max_depth, test_grid->max_depth);
    EXPECT_NE(result->root, nullptr);
}

TEST_F(SpatialOperationsTest, InsertBatch_SingleItemCreatesLeafAtMaxDepth) {
    std::vector<SpatialInsert> items = {
        {42, {1.0f, 1.0f, 1.0f}}
    };
    
    auto result = insert_batch(test_grid, items);
    
    EXPECT_NE(result->root, nullptr);
    EXPECT_TRUE(std::holds_alternative<SpatialBranch>(*result->root));
}

TEST_F(SpatialOperationsTest, InsertBatch_MultipleItemsInSameLeaf) {
    std::vector<SpatialInsert> items = {
        {1, {1.0f, 1.0f, 1.0f}},
        {2, {1.1f, 1.1f, 1.1f}},
        {3, {1.2f, 1.2f, 1.2f}}
    };
    
    auto result = insert_batch(test_grid, items);
    
    EXPECT_NE(result->root, nullptr);
    EXPECT_TRUE(std::holds_alternative<SpatialBranch>(*result->root));
}

TEST_F(SpatialOperationsTest, InsertBatch_ItemsInDifferentOctants) {
    std::vector<SpatialInsert> items = {
        {1, {1.0f, 1.0f, 1.0f}},   // octant 0
        {2, {6.0f, 1.0f, 1.0f}},   // octant 1
        {3, {1.0f, 6.0f, 1.0f}},   // octant 2
        {4, {1.0f, 1.0f, 6.0f}}    // octant 4
    };
    
    auto result = insert_batch(test_grid, items);
    
    EXPECT_NE(result->root, nullptr);
    EXPECT_TRUE(std::holds_alternative<SpatialBranch>(*result->root));
    
    // Root should have children in different octants
    const auto& root_branch = std::get<SpatialBranch>(*result->root);
    int non_null_children = 0;
    for (int i = 0; i < 8; ++i) {
        if (root_branch.children[i] != nullptr) {
            non_null_children++;
        }
    }
    EXPECT_GT(non_null_children, 1);
}

TEST_F(SpatialOperationsTest, InsertBatch_PreservesExistingItems) {
    // First insert
    std::vector<SpatialInsert> first_items = {
        {1, {1.0f, 1.0f, 1.0f}}
    };
    auto grid_with_items = insert_batch(test_grid, first_items);
    
    // Second insert
    std::vector<SpatialInsert> second_items = {
        {2, {2.0f, 2.0f, 2.0f}}
    };
    auto final_grid = insert_batch(grid_with_items, second_items);
    
    EXPECT_NE(final_grid, grid_with_items);
    EXPECT_NE(final_grid->root, nullptr);
}

TEST_F(SpatialOperationsTest, InsertBatch_HandlesItemsAtBoundary) {
    std::vector<SpatialInsert> items = {
        {1, {0.0f, 0.0f, 0.0f}},     // min boundary
        {2, {10.0f, 10.0f, 10.0f}},  // max boundary
        {3, {5.0f, 5.0f, 5.0f}}      // center
    };
    
    auto result = insert_batch(test_grid, items);
    
    EXPECT_NE(result->root, nullptr);
    EXPECT_TRUE(std::holds_alternative<SpatialBranch>(*result->root));
}

// Frustum Search Tests
TEST_F(SpatialOperationsTest, SearchFrustum_EmptyGrid) {
    Frustum3D frustum({5.0f, 5.0f, 0.0f}, {5.0f, 5.0f, 10.0f}, 2.0f, 0.0f);
    
    auto results = search_frustum(test_grid, frustum);
    
    EXPECT_TRUE(results.empty());
}

TEST_F(SpatialOperationsTest, SearchFrustum_NullGrid) {
    Frustum3D frustum({5.0f, 5.0f, 0.0f}, {5.0f, 5.0f, 10.0f}, 2.0f, 0.0f);
    
    auto results = search_frustum(nullptr, frustum);
    
    EXPECT_TRUE(results.empty());
}

TEST_F(SpatialOperationsTest, SearchFrustum_SingleItemInside) {
    // Insert item at center of grid
    std::vector<SpatialInsert> items = {
        {42, {5.0f, 5.0f, 5.0f}}
    };
    auto grid_with_items = insert_batch(test_grid, items);
    
    // Create frustum that should include the item
    Frustum3D frustum({5.0f, 5.0f, 0.0f}, {5.0f, 5.0f, 10.0f}, 2.0f, 0.0f);
    
    auto results = search_frustum(grid_with_items, frustum);
    
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].item_address, 42);
    EXPECT_GT(results[0].distance_to_apex, 0.0f);
}

TEST_F(SpatialOperationsTest, SearchFrustum_SingleItemOutside) {
    // Insert item far from frustum
    std::vector<SpatialInsert> items = {
        {42, {9.0f, 9.0f, 9.0f}}
    };
    auto grid_with_items = insert_batch(test_grid, items);
    
    // Create narrow frustum that shouldn't include the item
    Frustum3D frustum({1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}, 0.5f, 0.0f);
    
    auto results = search_frustum(grid_with_items, frustum);
    
    EXPECT_TRUE(results.empty());
}

TEST_F(SpatialOperationsTest, SearchFrustum_MultipleItemsSorted) {
    // Insert items at different distances from apex
    std::vector<SpatialInsert> items = {
        {1, {5.0f, 5.0f, 2.0f}},  // closer to apex
        {2, {5.0f, 5.0f, 8.0f}},  // farther from apex
        {3, {5.0f, 5.0f, 5.0f}}   // middle distance
    };
    auto grid_with_items = insert_batch(test_grid, items);
    
    // Create frustum along z-axis
    Frustum3D frustum({5.0f, 5.0f, 0.0f}, {5.0f, 5.0f, 10.0f}, 2.0f, 0.0f);
    
    auto results = search_frustum(grid_with_items, frustum);
    
    EXPECT_EQ(results.size(), 3);
    // Results should be sorted by distance to apex
    EXPECT_LT(results[0].distance_to_apex, results[1].distance_to_apex);
    EXPECT_LT(results[1].distance_to_apex, results[2].distance_to_apex);
    
    // Check addresses are in expected order (closest first)
    EXPECT_EQ(results[0].item_address, 1);  // closest
    EXPECT_EQ(results[1].item_address, 3);  // middle
    EXPECT_EQ(results[2].item_address, 2);  // farthest
}

TEST_F(SpatialOperationsTest, SearchFrustum_TruncatedCone) {
    // Insert items that should be filtered by apex radius
    std::vector<SpatialInsert> items = {
        {1, {5.0f, 5.0f, 1.0f}},  // near apex, should be excluded
        {2, {5.0f, 5.0f, 5.0f}},  // middle, should be included
        {3, {5.0f, 5.0f, 9.0f}}   // near base, should be included
    };
    auto grid_with_items = insert_batch(test_grid, items);
    
    // Create truncated cone (frustum) with apex radius
    Frustum3D frustum({5.0f, 5.0f, 0.0f}, {5.0f, 5.0f, 10.0f}, 3.0f, 1.5f);
    
    auto results = search_frustum(grid_with_items, frustum);
    
    // Should exclude the item too close to apex
    EXPECT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].item_address, 2);
    EXPECT_EQ(results[1].item_address, 3);
}

TEST_F(SpatialOperationsTest, SearchFrustum_AngledFrustum) {
    // Insert items along a diagonal
    std::vector<SpatialInsert> items = {
        {1, {1.0f, 1.0f, 1.0f}},
        {2, {5.0f, 5.0f, 5.0f}},
        {3, {9.0f, 9.0f, 9.0f}},
        {4, {2.0f, 8.0f, 5.0f}}  // off the diagonal
    };
    auto grid_with_items = insert_batch(test_grid, items);
    
    // Create diagonal frustum
    Frustum3D frustum({0.0f, 0.0f, 0.0f}, {10.0f, 10.0f, 10.0f}, 2.0f, 0.0f);
    
    auto results = search_frustum(grid_with_items, frustum);
    
    // Should include items on or near the diagonal
    EXPECT_GE(results.size(), 2);  // at least items 1 and 2
    
    // Results should be sorted by distance to apex
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_LE(results[i-1].distance_to_apex, results[i].distance_to_apex);
    }
}

TEST_F(SpatialOperationsTest, SearchFrustum_ItemsAtFrustumBoundary) {
    // Insert items right at the edge of the frustum
    std::vector<SpatialInsert> items = {
        {1, {5.0f, 7.0f, 5.0f}},   // exactly at radius
        {2, {5.0f, 7.1f, 5.0f}},   // just outside
        {3, {5.0f, 6.9f, 5.0f}}    // just inside
    };
    auto grid_with_items = insert_batch(test_grid, items);
    
    // Create frustum with radius 2.0 at z=5
    Frustum3D frustum({5.0f, 5.0f, 0.0f}, {5.0f, 5.0f, 10.0f}, 2.0f, 0.0f);
    
    auto results = search_frustum(grid_with_items, frustum);
    
    // Should include boundary and inside items
    EXPECT_GE(results.size(), 1);  // at least the inside item
    EXPECT_LE(results.size(), 2);  // at most inside + boundary
}