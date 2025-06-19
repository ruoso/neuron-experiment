# Design Document: Simulated World for Embodied Neural Creature

## Overview

This document defines the structure and dynamics of the simulated 2D world in which a neuron-driven creature exists. The environment is designed to be rich enough to support the emergence of nontrivial behaviors through self-organizing neural activity, while remaining efficient for simulation.

---

## 1. World Space

* **Type**: 2D continuous toroidal plane (wraparound space)
* **Dimensions**: Configurable (e.g., 100x100 units)
* **Units**: Arbitrary; interpreted as meters or gridless coordinates

### Properties:

* Movement and sensing are floating-point based
* Wraparound allows infinite-like exploration without borders

---

## 2. Objects in the World

### a. Trees

* Lifecycle: `Seedling → Mature → Fruiting → Dormant`
* Fruiting stage generates multiple fruit objects
* Trees can have timers or triggers for transitions

### b. Fruits

* **Properties**:

  * `Vec2 position`
  * `float maturity ∈ [0, 1]`
  * `float satiation_value = maturity * max_satiation`
  * `bool available`
* Ripen over time, possibly decay if not eaten
* Visible to creature via color-sensitive eye strips

### c. Obstacles (optional)

* Non-passable regions that block movement
* May attenuate visibility

### d. Terrain Zones (optional)

* Regions with different friction or movement costs (e.g., water, mud)

---

## 3. Environmental Dynamics

### Time Step

* Discrete ticks (e.g., 60 per second simulation rate)
* Each tick updates object states, maturity, etc.

### Tree/Fruit Update Rules

* Trees check age/lifecycle timer
* Fruits increase maturity per tick, capped at 1.0
* Fruits may decay or disappear after a timeout or if eaten

---

## 4. Interaction Rules

### Movement

* Creature moves via tank-style locomotion using left/right force differentials
* Movement is applied as velocity vector + angular rotation

### Food Interaction

* If within `eat_radius` of a fruit and 'eat' action is triggered:

  * Fruit is removed or marked unavailable
  * Creature's hunger is reduced
  * A spike is emitted to the creature’s brain to indicate satiety

---

## 5. Sensor Models (in-world reference)

### Vision

* Eye strips project rays across field of view
* Color is sampled per object within segment's arc
* Encoded as spike input events based on color and intensity

### Proprioception

* Motor output from the creature causes self-feedback spikes
* Also includes sensed forward velocity and turn rate

### Hunger

* Internal variable increasing over time
* Reduced when food is consumed
* Translates into hunger/satiety spike inputs

---

## 6. World API (for integration)

* `get_visible_objects(position, angle, fov)`
* `apply_motor_force(left, right)`
* `consume_if_in_range()`
* `get_internal_state()`
* `step_simulation(tick)`

---

## Goals

* Support embodied cognition through spatial, temporal, and motivational complexity
* Allow measurable emergent behavior via hunger satisfaction, movement optimization, and fruit discrimination
* Be extensible for more complex tasks in future iterations
