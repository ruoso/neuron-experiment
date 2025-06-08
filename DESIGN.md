# Design Document: 3D Spatial Neural Simulation with Dendritic Geometry and Event-Based Messaging

## Overview

This design outlines a biologically inspired neural simulation architecture, emphasizing spatial structure, dendritic computation, and efficient event-based spike messaging. The system models neurons embedded in 3D space, with fixed dendritic trees, polarity-guided axon/dendrite growth, and address-based messaging.

## Core Concepts

### Neuron Structure

Each neuron:

* Has a fixed 3D position in space
* Emits spikes to target dendritic terminals of other neurons
* Receives inputs through a structured dendritic tree
* Fires based on a thresholded sum of dendritic inputs

```cpp
struct Neuron {
    Vec3 position;
    float threshold;
    std::vector<uint32_t> output_targets; // Addresses of dendritic terminals
};
```

### Dendritic Tree and Addressing

Each neuron has a dendritic tree:

* Terminal and branch nodes are addressed using a hierarchical 32-bit scheme
* The last bits define leaf positions; higher levels are masked by zero bits
* Each dendritic node stores a weight for each of its inputs

```cpp
struct DendriticInput {
    uint32_t source_address;
    float weight;
};

struct DendriticBranch {
    std::vector<DendriticInput> inputs;
    float threshold;
    float last_output;
};
```

### Addressing Scheme

* 32-bit address space
* Terminal addresses are unique per neuron
* Masking zeros (e.g., last 3, 6, 12 bits) defines branches and soma
* Addresses are used to lookup weights and route spike messages

### Event-Based Messaging

* Spikes are binary (0/1)
* Processed in batches
* Only dendritic terminals and branches touched by messages need computation

```cpp
struct SpikeEvent {
    uint32_t target_address;
    float value; // Usually 1.0
};
```

### Weight Storage

* Global flat `float` array or hash map keyed by address
* Weights are stored only once per source→target edge
* No need for per-axon weights (axon terminals emit identical signals)

```cpp
std::unordered_map<uint32_t, float> weights;
```

### Temporal Decay (Optional)

* If used, store only the last activation time
* Decay computed on-demand during spike processing

```cpp
std::unordered_map<uint32_t, uint32_t> last_activation_time;
```

## Spatial Modeling

### Dendritic Terminal Position

* Each dendritic terminal stores its 3D position
* Position computed once based on flow field and branching structure

```cpp
struct DendriticTerminal {
    uint32_t address;
    Vec3 position;
};
```

### Flow Field

* Defines polarity/direction preference for axon and dendrite growth
* May be stored as a 3D vector grid or a function

```cpp
Vec3 flow_field[x][y][z];
```

### Axon Targeting

* Axons select dendritic terminals by spatial proximity and direction
* Use KD-tree, voxel grid, or region binning for fast lookup

### Wraparound Topology

* Space is toroidal or wraparound
* All coordinate math and spatial queries use modular arithmetic

## Simulation Loop

1. Collect spike events for the current tick
2. Group events by dendritic address
3. For each dendritic terminal/branch:

   * Lookup or compute decayed potential
   * Add weighted input from spike
   * If threshold exceeded, emit new spike from soma
4. Neurons that fire broadcast spike events to `output_targets`
5. Update timestamp map if using decay

## Summary of Design Benefits

| Feature                        | Benefit                                       |
| ------------------------------ | --------------------------------------------- |
| Stateless spike flow           | Fast, parallel, no per-tick potential storage |
| Address-driven weight lookup   | Compact, fast, and elegant structure          |
| Precomputed terminal positions | Enables directional and local targeting       |
| Optional decay via timestamps  | Lightweight temporal integration              |
| Fixed dendritic geometry       | Balanced realism and computational cost       |

This design enables large-scale, efficient, and spatially-aware neural simulations with localized learning, directional plasticity, and strong biological plausibility.

