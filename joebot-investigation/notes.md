# JoeBOT Neural Network Investigation - Research Notes

## Initial Research (2026-03-11)

### What is JoeBOT?

JoeBOT is a bot for Counter-Strike (the original Half-Life mod) created by Johannes Lampel (aka @$3.1415rin), active from roughly 2000-2005+. It's notable for being one of the earliest game bots to incorporate **artificial neural networks** into its AI decision-making.

- **Repository**: https://github.com/Bots-United/joebot
- **SourceForge**: https://sourceforge.net/projects/joebot/
- **License**: GPL v2
- **Language**: C++ (84.4%), C (9.7%)
- **Latest version**: 1.6.5.3 (February 2019 release, though active development was circa 2000-2005)
- **Dependencies**: Half-Life SDK 2.3, Metamod SDK 1.17

### Key Finding: Two Neural Networks

JoeBOT uses a **hybrid AI architecture** with two distinct feedforward neural networks trained via backpropagation:

1. **Combat Neural Network** - Decides tactical behavior during fights
2. **Collision Avoidance Neural Network** - Decides how to avoid obstacles

Everything else (navigation, objective completion) uses traditional waypoint-based pathfinding.

### The NNSim Library

JoeBOT includes its own custom neural network library called "NNSim" (by Johannes Lampel / LampelSoft). It implements:

- **Feedforward neural networks** with backpropagation learning (`CNeuralNetBProp`)
- **Backpropagation with momentum** (`CNeuralNetBPropM`)
- **Self-Organizing Maps (SOMs)** (`qSOM2d`) - 2D Kohonen maps
- **Genetic Algorithm support** (`CGenom`) with mutation and crossover operators
- **Activation functions**: tanh and logistic sigmoid, with lookup-table optimization for speed
- **Pattern management** for training data

### Combat Neural Network Architecture

From `NN_Train/main.cpp` and `NN_Train/combatnetdescription.txt`:

- **Type**: Feedforward, backpropagation
- **Topology**: 4 layers — `6 → 7 → 7 → 5`
  - Layer 0 (Input): 6 neurons
  - Layer 1 (Hidden): 7 neurons
  - Layer 2 (Hidden): 7 neurons
  - Layer 3 (Output): 5 neurons
- **Learning rate**: 0.1
- **Weight initialization**: [-0.3, 0.3]
- **Max training error target**: 0.2
- **Max training epochs per attempt**: 750 (reinitializes weights and retries if not converged)
- **Activation function**: tanh (with lookup table optimization in-game)

**Inputs (6):**
| Index | Name | Description | Range |
|-------|------|-------------|-------|
| 0 | IHealth | Bot's health level | -1 (low) to 1 (high) |
| 1 | IDistance | Distance to enemy | -1 (near) to 1 (far) |
| 2 | IEWeapon | Enemy weapon range | -1 (short) to 1 (long) |
| 3 | IWeapon | Bot's weapon range | -1 (short) to 1 (long) |
| 4 | IAmmo | Ammo in current clip | -1 (low) to 1 (high) |
| 5 | ISituation | Tactical situation | -1 (bad) to 1 (good) |

**Outputs (5):**
| Index | Name | Description | Values |
|-------|------|-------------|--------|
| 0 | ODuck | Should duck? | -1 (no) / 1 (yes) |
| 1 | OJump | Should jump? | -1 (no) / 1 (yes) |
| 2 | OHide | Should retreat/hide? | 0 (no) / 1 (yes) |
| 3 | OMoveType | Movement type | -1/0/1 (none/walk/run) |
| 4 | OStrafe | Strafe direction | -1 (left) / 0 (none) / 1 (right) |

**Training data**: Hand-crafted patterns encoding tactical knowledge, e.g.:
- High health + far distance + low enemy weapon + mid weapon + high ammo → run forward, no duck/jump
- Low health + near distance + low weapon → jump, strafe right, hide
- With sniper rifle at far distance → duck and hold
- Low ammo in bad situation → hide

About 30 base patterns are defined, then expanded programmatically (adding H_Mid variants of H_High patterns, adding S_Bad/S_Good situation variants), resulting in ~60-80 total training patterns.

### Collision Avoidance Neural Network Architecture

From `NN_Train/main.cpp` and `NN_Train/collnetdescription.txt`:

- **Topology**: 3 layers — `3 → 3 → 1`
  - Layer 0 (Input): 3 neurons
  - Layer 1 (Hidden): 3 neurons
  - Layer 2 (Output): 1 neuron
- **Learning rate**: 0.1
- **Max training error target**: 0.2
- **Max training epochs per attempt**: 1000

**Inputs (3):**
| Index | Name | Description |
|-------|------|-------------|
| 0 | ICI_Left | Obstacle distance left |
| 1 | ICI_Middle | Obstacle distance middle |
| 2 | ICI_Right | Obstacle distance right |

Input values are ray-trace fractions (0 = hit immediately, 1 = no hit), converted to [-1, 1] range and normalized as a vector.

**Output (1):**
| Value | Meaning |
|-------|---------|
| -1 | Steer left |
| 0 | Go straight |
| 1 | Steer right |

### In-Game Usage

**Combat NN** (`dlls/CBotCS_combat.cpp`):
- Called during `Fight()` method at a configurable update rate (`jb_nnupdaterate`)
- Input values are computed from live game state:
  - Health: `(bot_health / 50.0) - 1.0`
  - Distance: converted via `ConvertDistance()` function
  - Weapon values: looked up from per-weapon range rating table (e.g., AWP = 1.0, knife = -1.0)
  - Ammo: `(current_clip / max_clip) * 2.0 - 1.0`
  - Situation: `tanh((allies_fighting - enemies_fighting + manner + aggressivity) / 2)`
- NN output directly controls bot ducking, jumping, hiding, movement type, and strafing

**Collision NN** (`dlls/CBotBase.cpp`):
- Called during navigation when any of 3 ray traces (left, middle, right) detect an obstacle
- Output > 0.5: strafe right; Output < -0.5: strafe left; otherwise: go straight

### SOM (Self-Organizing Map) Analysis

The codebase includes SOM analysis tools (`SOM_Analysis_LIN`, `SOM_Analysis_Win`) and the SOM was apparently used to analyze combat input patterns during gameplay (see commented-out code: `SP.AddPattern(dCombatNNIn)`). This was likely used to visualize and cluster the kinds of combat situations the bot encountered, potentially for refining training data.

### Genetic Algorithm Support

The `CGenom` class supports:
- Extracting topology and weights from a trained network
- Mutation with configurable probability and range
- Crossover at a specified cut point
- Save/load of genome data (topology, weights, biases as separate files)

This suggests the developer experimented with evolutionary optimization of the neural networks, though the primary training method used in the shipped version is backpropagation.

### Performance Optimization: Activation Lookup Tables

A clever optimization: rather than computing `tanh()` or `sigmoid()` for every neuron on every propagation, NNSim pre-computes a lookup table at startup covering the range [-3, 3] with 100 samples per unit (600 total entries). The runtime activation functions do a table lookup instead of calling `math.h` functions. This was important for running inside a real-time game engine in the early 2000s.

### Weapon Range Ratings

Each CS weapon has a pre-defined "range" value from -1 (close range) to +1 (long range):
- AWP: 1.0 (sniper)
- Scout/G3SG1/SG550: 0.9
- AUG/M4A1: 0.6
- AK47/Deagle: 0.0
- Knife: -1.0
- USP: -0.6

### What's Interesting About This Project

1. **Early neural network game AI** (2000-2002): This predates most ML-in-games work by over a decade
2. **Hand-crafted training data**: Unlike modern approaches, the neural nets were trained on manually designed scenarios encoding expert tactical knowledge
3. **Hybrid architecture**: Neural nets handle only specific subsystems (combat tactics, collision avoidance); everything else is traditional rule-based/waypoint code
4. **Custom NN library**: The developer wrote the entire neural network framework from scratch in C++
5. **SOM for analysis**: Self-organizing maps were used as an analytical tool to understand what situations the bot encountered
6. **Genetic algorithm experiments**: Infrastructure for evolving networks was built, suggesting experimentation beyond pure backpropagation
7. **Performance-conscious**: Lookup tables for activation functions show awareness of real-time constraints
8. **The networks are tiny**: 6→7→7→5 and 3→3→1 — extremely small by modern standards, but effective for their purpose
