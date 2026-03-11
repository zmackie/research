# JoeBOT: Reverse-Engineering a Neural Network Game Bot from 2000

An investigation into **JoeBOT**, one of the earliest game bots to use artificial neural networks, created for Counter-Strike by Johannes Lampel ([@$3.1415rin](http://joebot.bots-united.com/)) between 2000 and 2005.

## What is JoeBOT?

JoeBOT is an AI bot for the original Counter-Strike (a Half-Life mod). While most bots of that era relied entirely on scripted rules and waypoint navigation, JoeBOT was notable for incorporating **feedforward neural networks trained with backpropagation** into its combat and collision avoidance systems. The source code is [available on GitHub](https://github.com/Bots-United/joebot) under the GPL v2 license.

## Architecture: A Hybrid AI System

JoeBOT uses a hybrid approach: traditional waypoint-based pathfinding for navigation, combined with two small neural networks for real-time tactical decisions.

### Combat Neural Network (6 → 7 → 7 → 5)

The combat NN decides how the bot should behave during a fight. It's a 4-layer feedforward network with **151 total parameters** (126 weights + 25 biases), using tanh activation.

**Inputs (6 neurons):**
| Input | Description | Range |
|-------|------------|-------|
| Health | Bot's health | -1 (dying) to +1 (full) |
| Distance | Range to enemy | -1 (point blank) to +1 (far) |
| Enemy Weapon | Enemy's weapon type | -1 (knife) to +1 (AWP) |
| Weapon | Bot's weapon type | -1 (knife) to +1 (AWP) |
| Ammo | Current clip level | -1 (empty) to +1 (full) |
| Situation | Tactical advantage | -1 (outnumbered) to +1 (advantage) |

**Outputs (5 neurons):**
| Output | Description | Interpretation |
|--------|------------|----------------|
| Duck | Crouch | > 0 = duck |
| Jump | Jump | > 0 = jump |
| Hide | Retreat | > 0.5 = seek cover |
| Move Type | Speed | -1/0/+1 = stop/walk/run |
| Strafe | Lateral movement | -1/0/+1 = left/none/right |

### Collision Avoidance Neural Network (3 → 3 → 1)

The collision NN handles obstacle avoidance with just **19 parameters**. It takes three ray-trace distances (left, middle, right) and outputs a steering direction.

| Input | Description |
|-------|------------|
| Left | Distance to obstacle on the left |
| Middle | Distance to obstacle ahead |
| Right | Distance to obstacle on the right |

Output: -1 (steer left) to +1 (steer right), with a dead zone around 0 (go straight).

## Sample Outputs

Running the re-implemented networks on various scenarios:

```
Scenario: Healthy sniper vs weak enemy at distance
  → DUCK | no jump | engage | run | no strafe
  (Sniper crouches for accuracy, holds position)

Scenario: Low health, close range, shotgun vs sniper, outnumbered
  → stand | JUMP | HIDE/RETREAT | run | strafe right
  (Desperate evasion — jump, strafe, seek cover)

Scenario: Low health, knife fight, outnumbered
  → stand | JUMP | HIDE/RETREAT | run | strafe right
  (All evasion outputs maxed — pure survival mode)

Scenario: AWP vs AWP at distance, good situation
  → DUCK | no jump | engage | run | no strafe
  (Classic sniper duel behavior — crouch and hold)
```

The distance-vs-health sweep reveals the network learned graduated responses:

- **Close + low health** → jump + strafe + hide (full panic mode)
- **Close + high health** → strafe only (aggressive close combat)
- **Far + low health** → hide (retreat to safety)
- **Far + high health** → stand and shoot (confident engagement)

## Training Methodology

The training is remarkable by modern standards: **entirely hand-crafted patterns**. The developer manually defined ~30 base combat scenarios encoding tactical knowledge (e.g., "if health is low and enemy has a sniper at close range, jump and retreat"). These were then programmatically expanded to ~60-80 patterns by:

1. Adding medium-health variants of all high-health scenarios
2. Generating "bad situation" variants where bots should hide more
3. Generating "good situation" variants where bots can be more aggressive

Training uses standard backpropagation with learning rate 0.1, weight initialization in [-0.3, 0.3], and a maximum error threshold of 0.2. If training doesn't converge within 750 epochs, weights are re-initialized and training restarts.

## The NNSim Library

The developer wrote a complete neural network framework from scratch in C++, called **NNSim** (by "LampelSoft"). It includes:

- **Feedforward networks** with backpropagation (`CNeuralNetBProp`)
- **Backpropagation with momentum** (`CNeuralNetBPropM`)
- **Self-Organizing Maps** (`qSOM2d`) — 2D Kohonen maps for analyzing combat patterns
- **Genetic algorithms** (`CGenom`) — mutation, crossover, and genome serialization
- **Lookup-table activation functions** — pre-computed tanh/sigmoid tables for real-time performance (range [-3, 3] with 600 entries)

## What Makes This Interesting

1. **Historical significance**: This is one of the earliest examples of neural networks in game AI (2000-2002), predating the deep learning revolution by over a decade.

2. **Expert-system meets neural nets**: The training data was hand-crafted expert knowledge, not learned from gameplay data. The neural network serves as a differentiable interpolation of human tactical expertise — it generalizes the hand-designed patterns to handle intermediate situations.

3. **Tiny but effective**: The combat network has just 151 parameters. For comparison, GPT-2 has 1.5 billion. Yet this tiny network produces plausible tactical behavior.

4. **The right tool for the job**: The developer recognized that neural nets aren't needed for everything. Navigation uses waypoints, buying uses rules, and only combat/collision use NNs — a pragmatic hybrid that works.

5. **SOM analysis tooling**: The inclusion of Self-Organizing Map analysis tools suggests the developer used unsupervised learning to understand what combat situations the bot encountered in practice, potentially guiding the design of training patterns.

6. **Genetic algorithm infrastructure**: The genome serialization and mutation/crossover support suggests experiments with evolving the networks, though the shipped version uses backpropagation.

## Files in This Investigation

- `README.md` — This report
- `notes.md` — Detailed research notes with all technical findings
- `joebot_nn.py` — Python re-implementation of both neural networks with the trained weights, scenario testing, and analysis
- `joebot_nn_data.json` — Exported network weights/biases in JSON format for external analysis

## Sources

- [Bots-United/joebot on GitHub](https://github.com/Bots-United/joebot) — Full source code
- [JoeBOT on SourceForge](https://sourceforge.net/projects/joebot/) — Original distribution
- [JoeBOT homepage](http://joebot.bots-united.com/) — Original project page (archived)
