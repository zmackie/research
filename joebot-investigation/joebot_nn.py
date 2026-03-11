#!/usr/bin/env python3
"""
JoeBOT Neural Network Re-implementation in Python

Recreates the two neural networks used by JoeBOT (a Counter-Strike bot from ~2000-2002)
by parsing the original network description files and implementing forward propagation.

The original C++ code used a custom library called "NNSim" by Johannes Lampel.
This Python version demonstrates how the networks work and visualizes their behavior.

Source: https://github.com/Bots-United/joebot
"""

import math
import json
from dataclasses import dataclass


# ============================================================================
# Neural Network definitions from BotNNDefs.h
# ============================================================================

# Combat NN Input indices
I_HEALTH = 0      # Bot health: -1 (low) to 1 (high)
I_DISTANCE = 1    # Distance to enemy: -1 (near) to 1 (far)
I_EWEAPON = 2     # Enemy weapon range: -1 (short) to 1 (long)
I_WEAPON = 3      # Bot weapon range: -1 (short) to 1 (long)
I_AMMO = 4        # Ammo in clip: -1 (low) to 1 (high)
I_SITUATION = 5   # Tactical situation: -1 (bad) to 1 (good)
NUM_COMBAT_INPUTS = 6

# Combat NN Output indices
O_DUCK = 0        # Duck: -1 (no) / 1 (yes)
O_JUMP = 1        # Jump: -1 (no) / 1 (yes)
O_HIDE = 2        # Hide/retreat: 0 (no) / 1 (yes)
O_MOVETYPE = 3    # Move type: -1 (none) / 0 (walk) / 1 (run)
O_STRAFE = 4      # Strafe: -1 (left) / 0 (none) / 1 (right)
NUM_COMBAT_OUTPUTS = 5

# Collision NN Input indices
ICI_LEFT = 0
ICI_MIDDLE = 1
ICI_RIGHT = 2

# Weapon range ratings from BotNNDefs.h
WEAPON_RANGES = {
    "ak47": 0.0, "aug": 0.6, "awp": 1.0, "c4": -0.9,
    "deagle": 0.0, "elite": 0.1, "flashbang": -0.5,
    "g3sg1": 0.9, "glock18": -0.5, "hegrenade": -0.5,
    "knife": -1.0, "m249": -0.3, "m3": -0.2, "m4a1": 0.6,
    "mac10": -0.1, "mp5navy": 0.2, "p228": -0.2, "p90": 0.1,
    "tmp": 0.3, "usp": -0.6, "scout": 0.9, "sg552": 0.4,
    "smokegrenade": -0.5, "xm1014": -0.4, "fiveseven": -0.2,
    "sg550": 0.9, "ump45": 0.4, "galil": 0.0, "famas": 0.1,
    "shield": -1.0,
}


@dataclass
class FeedforwardNet:
    """A simple feedforward neural network with tanh activation."""
    layers: list          # neurons per layer
    weights: list         # weights[layer] = 2D list [from_neuron][to_neuron]
    biases: list          # biases[layer] = list of bias values per neuron

    def propagate(self, inputs: list) -> list:
        """Forward propagation through the network."""
        activations = list(inputs)
        for layer_idx in range(len(self.layers) - 1):
            next_size = self.layers[layer_idx + 1]
            next_activations = []
            for j in range(next_size):
                net = 0.0
                for i, act in enumerate(activations):
                    net += self.weights[layer_idx][i][j] * act
                # Apply bias and tanh activation
                bias = self.biases[layer_idx + 1][j]
                net_val = net - bias
                next_activations.append(math.tanh(net_val))
            activations = next_activations
        return activations


def parse_net_description(text: str) -> FeedforwardNet:
    """Parse a JoeBOT network description file into a FeedforwardNet."""
    lines = text.strip().split("\n")
    layers = []
    weights = {}
    biases = {}

    for line in lines:
        line = line.strip()
        if line.startswith("with") and "layers" in line:
            num_layers = int(line.split()[1])
        elif "neurons on layer" in line:
            parts = line.split()
            n_neurons = int(parts[0])
            layer_idx = int(parts[-1].replace("#", ""))
            layers.append(n_neurons)
            biases[layer_idx] = [0.0] * n_neurons
        elif line.startswith("Set the bias"):
            parts = line.split()
            neuron_idx = int(parts[5])
            layer_idx = int(parts[8])
            bias_val = float(parts[10])
            biases[layer_idx][neuron_idx] = bias_val
        elif line.startswith("Connect neuron"):
            parts = line.split()
            from_neuron = int(parts[2])
            from_layer = int(parts[5])
            to_neuron = int(parts[8])
            to_layer = int(parts[11])
            weight = float(parts[13])
            key = from_layer
            if key not in weights:
                weights[key] = {}
            if from_neuron not in weights[key]:
                weights[key][from_neuron] = {}
            weights[key][from_neuron][to_neuron] = weight

    # Convert to list structure
    weight_list = []
    for l in range(len(layers) - 1):
        w = []
        for i in range(layers[l]):
            row = []
            for j in range(layers[l + 1]):
                row.append(weights.get(l, {}).get(i, {}).get(j, 0.0))
            w.append(row)
        weight_list.append(w)

    bias_list = [biases.get(l, [0.0] * layers[l]) for l in range(len(layers))]

    return FeedforwardNet(layers=layers, weights=weight_list, biases=bias_list)


# ============================================================================
# Load the trained networks from the description files
# ============================================================================

COMBAT_NET_DESC = """Create a Feedforward NeuralNet
with 4 layers
6 neurons on layer #0
7 neurons on layer #1
7 neurons on layer #2
5 neurons on layer #3
and that's all about the basic topology
Set the bias of neuron 0 on layer 0 to 0.000000
Connect neuron 0 on layer 0 with neuron 0 on layer 1 with 0.643483
Connect neuron 0 on layer 0 with neuron 1 on layer 1 with -0.438803
Connect neuron 0 on layer 0 with neuron 2 on layer 1 with 2.440910
Connect neuron 0 on layer 0 with neuron 3 on layer 1 with -0.832760
Connect neuron 0 on layer 0 with neuron 4 on layer 1 with 0.331288
Connect neuron 0 on layer 0 with neuron 5 on layer 1 with -0.516746
Connect neuron 0 on layer 0 with neuron 6 on layer 1 with 0.825257
Set the bias of neuron 1 on layer 0 to 0.000000
Connect neuron 1 on layer 0 with neuron 0 on layer 1 with 1.579590
Connect neuron 1 on layer 0 with neuron 1 on layer 1 with -0.064085
Connect neuron 1 on layer 0 with neuron 2 on layer 1 with 2.204689
Connect neuron 1 on layer 0 with neuron 3 on layer 1 with 1.736021
Connect neuron 1 on layer 0 with neuron 4 on layer 1 with -1.306171
Connect neuron 1 on layer 0 with neuron 5 on layer 1 with 0.251432
Connect neuron 1 on layer 0 with neuron 6 on layer 1 with 0.876429
Set the bias of neuron 2 on layer 0 to 0.000000
Connect neuron 2 on layer 0 with neuron 0 on layer 1 with -0.154797
Connect neuron 2 on layer 0 with neuron 1 on layer 1 with 0.066674
Connect neuron 2 on layer 0 with neuron 2 on layer 1 with 0.156705
Connect neuron 2 on layer 0 with neuron 3 on layer 1 with 0.431899
Connect neuron 2 on layer 0 with neuron 4 on layer 1 with -0.774722
Connect neuron 2 on layer 0 with neuron 5 on layer 1 with -0.014679
Connect neuron 2 on layer 0 with neuron 6 on layer 1 with -0.232102
Set the bias of neuron 3 on layer 0 to 0.000000
Connect neuron 3 on layer 0 with neuron 0 on layer 1 with 0.769536
Connect neuron 3 on layer 0 with neuron 1 on layer 1 with 0.279660
Connect neuron 3 on layer 0 with neuron 2 on layer 1 with 2.322134
Connect neuron 3 on layer 0 with neuron 3 on layer 1 with 0.829340
Connect neuron 3 on layer 0 with neuron 4 on layer 1 with -0.968053
Connect neuron 3 on layer 0 with neuron 5 on layer 1 with 1.478761
Connect neuron 3 on layer 0 with neuron 6 on layer 1 with 1.713881
Set the bias of neuron 4 on layer 0 to 0.000000
Connect neuron 4 on layer 0 with neuron 0 on layer 1 with -0.344050
Connect neuron 4 on layer 0 with neuron 1 on layer 1 with 1.325939
Connect neuron 4 on layer 0 with neuron 2 on layer 1 with 0.870588
Connect neuron 4 on layer 0 with neuron 3 on layer 1 with 0.019880
Connect neuron 4 on layer 0 with neuron 4 on layer 1 with -0.098816
Connect neuron 4 on layer 0 with neuron 5 on layer 1 with -0.319643
Connect neuron 4 on layer 0 with neuron 6 on layer 1 with -0.386322
Set the bias of neuron 5 on layer 0 to 0.000000
Connect neuron 5 on layer 0 with neuron 0 on layer 1 with 0.290890
Connect neuron 5 on layer 0 with neuron 1 on layer 1 with 2.041458
Connect neuron 5 on layer 0 with neuron 2 on layer 1 with -0.152264
Connect neuron 5 on layer 0 with neuron 3 on layer 1 with -0.447083
Connect neuron 5 on layer 0 with neuron 4 on layer 1 with 1.380196
Connect neuron 5 on layer 0 with neuron 5 on layer 1 with 0.011982
Connect neuron 5 on layer 0 with neuron 6 on layer 1 with 1.163884
Set the bias of neuron 0 on layer 1 to -1.053993
Connect neuron 0 on layer 1 with neuron 0 on layer 2 with 0.522600
Connect neuron 0 on layer 1 with neuron 1 on layer 2 with -0.090600
Connect neuron 0 on layer 1 with neuron 2 on layer 2 with -0.087435
Connect neuron 0 on layer 1 with neuron 3 on layer 2 with -0.925527
Connect neuron 0 on layer 1 with neuron 4 on layer 2 with 1.778258
Connect neuron 0 on layer 1 with neuron 5 on layer 2 with 1.557704
Connect neuron 0 on layer 1 with neuron 6 on layer 2 with 1.232218
Set the bias of neuron 1 on layer 1 to -0.137965
Connect neuron 1 on layer 1 with neuron 0 on layer 2 with 2.763488
Connect neuron 1 on layer 1 with neuron 1 on layer 2 with 0.076769
Connect neuron 1 on layer 1 with neuron 2 on layer 2 with 0.028539
Connect neuron 1 on layer 1 with neuron 3 on layer 2 with 0.361922
Connect neuron 1 on layer 1 with neuron 4 on layer 2 with 0.082033
Connect neuron 1 on layer 1 with neuron 5 on layer 2 with -0.494710
Connect neuron 1 on layer 1 with neuron 6 on layer 2 with 0.173645
Set the bias of neuron 2 on layer 1 to -2.306823
Connect neuron 2 on layer 1 with neuron 0 on layer 2 with 0.140262
Connect neuron 2 on layer 1 with neuron 1 on layer 2 with -0.269354
Connect neuron 2 on layer 1 with neuron 2 on layer 2 with -0.437226
Connect neuron 2 on layer 1 with neuron 3 on layer 2 with -1.297921
Connect neuron 2 on layer 1 with neuron 4 on layer 2 with 1.292026
Connect neuron 2 on layer 1 with neuron 5 on layer 2 with 0.975024
Connect neuron 2 on layer 1 with neuron 6 on layer 2 with 0.829968
Set the bias of neuron 3 on layer 1 to -1.657151
Connect neuron 3 on layer 1 with neuron 0 on layer 2 with -0.856214
Connect neuron 3 on layer 1 with neuron 1 on layer 2 with 0.162658
Connect neuron 3 on layer 1 with neuron 2 on layer 2 with 0.397890
Connect neuron 3 on layer 1 with neuron 3 on layer 2 with 0.251016
Connect neuron 3 on layer 1 with neuron 4 on layer 2 with 0.145386
Connect neuron 3 on layer 1 with neuron 5 on layer 2 with 0.744856
Connect neuron 3 on layer 1 with neuron 6 on layer 2 with 1.743528
Set the bias of neuron 4 on layer 1 to -0.612222
Connect neuron 4 on layer 1 with neuron 0 on layer 2 with 1.774877
Connect neuron 4 on layer 1 with neuron 1 on layer 2 with -0.509286
Connect neuron 4 on layer 1 with neuron 2 on layer 2 with -0.178570
Connect neuron 4 on layer 1 with neuron 3 on layer 2 with 0.137081
Connect neuron 4 on layer 1 with neuron 4 on layer 2 with -0.310092
Connect neuron 4 on layer 1 with neuron 5 on layer 2 with -0.487727
Connect neuron 4 on layer 1 with neuron 6 on layer 2 with 0.516750
Set the bias of neuron 5 on layer 1 to 0.855265
Connect neuron 5 on layer 1 with neuron 0 on layer 2 with -0.005108
Connect neuron 5 on layer 1 with neuron 1 on layer 2 with 1.511990
Connect neuron 5 on layer 1 with neuron 2 on layer 2 with 1.394428
Connect neuron 5 on layer 1 with neuron 3 on layer 2 with 0.515786
Connect neuron 5 on layer 1 with neuron 4 on layer 2 with -0.444844
Connect neuron 5 on layer 1 with neuron 5 on layer 2 with -0.534950
Connect neuron 5 on layer 1 with neuron 6 on layer 2 with 0.869386
Set the bias of neuron 6 on layer 1 to 0.697880
Connect neuron 6 on layer 1 with neuron 0 on layer 2 with 1.777144
Connect neuron 6 on layer 1 with neuron 1 on layer 2 with 0.994186
Connect neuron 6 on layer 1 with neuron 2 on layer 2 with 0.861439
Connect neuron 6 on layer 1 with neuron 3 on layer 2 with -0.367715
Connect neuron 6 on layer 1 with neuron 4 on layer 2 with 0.356446
Connect neuron 6 on layer 1 with neuron 5 on layer 2 with 0.067684
Connect neuron 6 on layer 1 with neuron 6 on layer 2 with 0.468132
Set the bias of neuron 0 on layer 2 to 0.006245
Connect neuron 0 on layer 2 with neuron 0 on layer 3 with 0.091429
Connect neuron 0 on layer 2 with neuron 1 on layer 3 with -0.284534
Connect neuron 0 on layer 2 with neuron 2 on layer 3 with -2.235371
Connect neuron 0 on layer 2 with neuron 3 on layer 3 with -0.030574
Connect neuron 0 on layer 2 with neuron 4 on layer 3 with 0.009112
Set the bias of neuron 1 on layer 2 to 1.317481
Connect neuron 1 on layer 2 with neuron 0 on layer 3 with 2.082160
Connect neuron 1 on layer 2 with neuron 1 on layer 3 with 0.641649
Connect neuron 1 on layer 2 with neuron 2 on layer 3 with -0.066009
Connect neuron 1 on layer 2 with neuron 3 on layer 3 with -1.142241
Connect neuron 1 on layer 2 with neuron 4 on layer 3 with 0.255443
Set the bias of neuron 2 on layer 2 to 0.380430
Connect neuron 2 on layer 2 with neuron 0 on layer 3 with 1.985437
Connect neuron 2 on layer 2 with neuron 1 on layer 3 with 0.806073
Connect neuron 2 on layer 2 with neuron 2 on layer 3 with 0.080591
Connect neuron 2 on layer 2 with neuron 3 on layer 3 with 0.145999
Connect neuron 2 on layer 2 with neuron 4 on layer 3 with -0.248217
Set the bias of neuron 3 on layer 2 to -1.158759
Connect neuron 3 on layer 2 with neuron 0 on layer 3 with -1.866667
Connect neuron 3 on layer 2 with neuron 1 on layer 3 with 0.949617
Connect neuron 3 on layer 2 with neuron 2 on layer 3 with -0.266451
Connect neuron 3 on layer 2 with neuron 3 on layer 3 with 0.657205
Connect neuron 3 on layer 2 with neuron 4 on layer 3 with -0.079026
Set the bias of neuron 4 on layer 2 to 0.730198
Connect neuron 4 on layer 2 with neuron 0 on layer 3 with 0.889667
Connect neuron 4 on layer 2 with neuron 1 on layer 3 with -1.756492
Connect neuron 4 on layer 2 with neuron 2 on layer 3 with -0.203640
Connect neuron 4 on layer 2 with neuron 3 on layer 3 with -0.079732
Connect neuron 4 on layer 2 with neuron 4 on layer 3 with -0.130964
Set the bias of neuron 5 on layer 2 to 0.562320
Connect neuron 5 on layer 2 with neuron 0 on layer 3 with 0.745888
Connect neuron 5 on layer 2 with neuron 1 on layer 3 with -2.154327
Connect neuron 5 on layer 2 with neuron 2 on layer 3 with -0.018323
Connect neuron 5 on layer 2 with neuron 3 on layer 3 with 0.204864
Connect neuron 5 on layer 2 with neuron 4 on layer 3 with -0.043539
Set the bias of neuron 6 on layer 2 to -1.993306
Connect neuron 6 on layer 2 with neuron 0 on layer 3 with -0.995368
Connect neuron 6 on layer 2 with neuron 1 on layer 3 with -0.968286
Connect neuron 6 on layer 2 with neuron 2 on layer 3 with 0.026060
Connect neuron 6 on layer 2 with neuron 3 on layer 3 with 1.344358
Connect neuron 6 on layer 2 with neuron 4 on layer 3 with -2.446062
Set the bias of neuron 0 on layer 3 to 1.687720
Set the bias of neuron 1 on layer 3 to 0.164209
Set the bias of neuron 2 on layer 3 to -2.216190
Set the bias of neuron 3 on layer 3 to -3.896950
Set the bias of neuron 4 on layer 3 to -2.591711
and that was the whole net :D"""

COLLISION_NET_DESC = """Create a Feedforward NeuralNet
with 3 layers
3 neurons on layer #0
3 neurons on layer #1
1 neurons on layer #2
and that's all about the basic topology
Set the bias of neuron 0 on layer 0 to 0.000000
Connect neuron 0 on layer 0 with neuron 0 on layer 1 with 0.089985
Connect neuron 0 on layer 0 with neuron 1 on layer 1 with 2.160398
Connect neuron 0 on layer 0 with neuron 2 on layer 1 with -0.187618
Set the bias of neuron 1 on layer 0 to 0.000000
Connect neuron 1 on layer 0 with neuron 0 on layer 1 with -0.570984
Connect neuron 1 on layer 0 with neuron 1 on layer 1 with 0.715427
Connect neuron 1 on layer 0 with neuron 2 on layer 1 with -0.612794
Set the bias of neuron 2 on layer 0 to 0.000000
Connect neuron 2 on layer 0 with neuron 0 on layer 1 with 1.806571
Connect neuron 2 on layer 0 with neuron 1 on layer 1 with -0.157420
Connect neuron 2 on layer 0 with neuron 2 on layer 1 with 2.155664
Set the bias of neuron 0 on layer 1 to 1.274111
Connect neuron 0 on layer 1 with neuron 0 on layer 2 with 0.935178
Set the bias of neuron 1 on layer 1 to 0.777270
Connect neuron 1 on layer 1 with neuron 0 on layer 2 with -1.393603
Set the bias of neuron 2 on layer 1 to 0.739181
Connect neuron 2 on layer 1 with neuron 0 on layer 2 with 1.723929
Set the bias of neuron 0 on layer 2 to -1.184349
and that was the whole net :D"""


def interpret_combat_output(outputs):
    """Interpret the raw combat NN output into human-readable decisions."""
    duck = "DUCK" if outputs[O_DUCK] > 0 else "stand"
    jump = "JUMP" if outputs[O_JUMP] > 0 else "no jump"
    hide = "HIDE/RETREAT" if outputs[O_HIDE] > 0.5 else "engage"
    if outputs[O_MOVETYPE] > 0.33:
        move = "run"
    elif outputs[O_MOVETYPE] > -0.33:
        move = "walk"
    else:
        move = "stop"
    if outputs[O_STRAFE] > 0.33:
        strafe = "right"
    elif outputs[O_STRAFE] < -0.33:
        strafe = "left"
    else:
        strafe = "none"
    return {
        "duck": duck, "jump": jump, "hide": hide,
        "move": move, "strafe": strafe,
        "raw": {
            "duck": round(outputs[O_DUCK], 3),
            "jump": round(outputs[O_JUMP], 3),
            "hide": round(outputs[O_HIDE], 3),
            "movetype": round(outputs[O_MOVETYPE], 3),
            "strafe": round(outputs[O_STRAFE], 3),
        }
    }


def interpret_collision_output(output_val):
    """Interpret the collision NN output."""
    if output_val > 0.5:
        return "steer RIGHT"
    elif output_val < -0.5:
        return "steer LEFT"
    else:
        return "go STRAIGHT"


def normalize_vec(v):
    """Normalize a vector (as done in the collision NN training)."""
    mag = math.sqrt(sum(x * x for x in v))
    if mag == 0:
        return v
    return [x / mag for x in v]


def run_demo():
    """Run a demonstration of both neural networks."""
    print("=" * 72)
    print("JoeBOT Neural Network Re-implementation")
    print("Original by Johannes Lampel (@$3.1415rin), circa 2000-2002")
    print("Python re-implementation for analysis, 2026")
    print("=" * 72)

    # Parse networks
    combat_nn = parse_net_description(COMBAT_NET_DESC)
    collision_nn = parse_net_description(COLLISION_NET_DESC)

    print(f"\nCombat NN topology: {' -> '.join(str(n) for n in combat_nn.layers)}")
    print(f"  Total weights: {sum(combat_nn.layers[i] * combat_nn.layers[i+1] for i in range(len(combat_nn.layers)-1))}")
    print(f"  Total biases: {sum(combat_nn.layers)}")
    total_params = (sum(combat_nn.layers[i] * combat_nn.layers[i+1] for i in range(len(combat_nn.layers)-1))
                    + sum(combat_nn.layers))
    print(f"  Total parameters: {total_params}")

    print(f"\nCollision NN topology: {' -> '.join(str(n) for n in collision_nn.layers)}")
    coll_weights = sum(collision_nn.layers[i] * collision_nn.layers[i+1] for i in range(len(collision_nn.layers)-1))
    coll_biases = sum(collision_nn.layers)
    print(f"  Total weights: {coll_weights}")
    print(f"  Total biases: {coll_biases}")
    print(f"  Total parameters: {coll_weights + coll_biases}")

    # -----------------------------------------------------------------------
    # Combat NN scenarios
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("COMBAT NEURAL NETWORK SCENARIOS")
    print("=" * 72)

    scenarios = [
        {
            "name": "Healthy sniper vs weak enemy at distance",
            "inputs": [1.0, 1.0, -1.0, 1.0, 1.0, 0.0],
            "desc": "health=high, dist=far, eweapon=low, weapon=sniper, ammo=high, situation=normal"
        },
        {
            "name": "Low health, close range, shotgun vs sniper",
            "inputs": [-1.0, -1.0, 1.0, -0.2, 0.5, -1.0],
            "desc": "health=low, dist=near, eweapon=sniper, weapon=m3, ammo=mid, situation=bad"
        },
        {
            "name": "Full health, AK47, normal fight",
            "inputs": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "desc": "health=high, dist=mid, eweapon=mid, weapon=ak47, ammo=high, situation=normal"
        },
        {
            "name": "Low health, knife fight, outnumbered",
            "inputs": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            "desc": "health=low, dist=near, eweapon=low, weapon=knife, ammo=none, situation=bad"
        },
        {
            "name": "AWP vs AWP at distance, good situation",
            "inputs": [0.5, 1.0, 1.0, 1.0, 0.5, 1.0],
            "desc": "health=mid, dist=far, eweapon=sniper, weapon=awp, ammo=mid, situation=good"
        },
        {
            "name": "Rushing with P90, close range",
            "inputs": [1.0, -1.0, -0.5, 0.1, 1.0, 1.0],
            "desc": "health=high, dist=near, eweapon=glock, weapon=p90, ammo=full, situation=good"
        },
    ]

    for i, scenario in enumerate(scenarios):
        outputs = combat_nn.propagate(scenario["inputs"])
        decision = interpret_combat_output(outputs)
        print(f"\n--- Scenario {i+1}: {scenario['name']} ---")
        print(f"  {scenario['desc']}")
        print(f"  Decision: {decision['duck']} | {decision['jump']} | {decision['hide']} | move={decision['move']} | strafe={decision['strafe']}")
        print(f"  Raw outputs: {decision['raw']}")

    # -----------------------------------------------------------------------
    # Collision NN scenarios
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("COLLISION AVOIDANCE NEURAL NETWORK SCENARIOS")
    print("=" * 72)

    coll_scenarios = [
        {"name": "Wall on left", "inputs": [1.0, -1.0, -1.0]},
        {"name": "Wall on right", "inputs": [-1.0, -1.0, 1.0]},
        {"name": "Wall ahead", "inputs": [0.0, -1.0, 0.0]},
        {"name": "Clear path", "inputs": [1.0, 1.0, 1.0]},
        {"name": "Narrow corridor, slightly left", "inputs": [0.0, 0.5, -0.5]},
        {"name": "Wall ahead and right", "inputs": [-1.0, -1.0, 1.0]},
    ]

    for i, scenario in enumerate(coll_scenarios):
        normalized = normalize_vec(scenario["inputs"])
        outputs = collision_nn.propagate(normalized)
        decision = interpret_collision_output(outputs[0])
        print(f"\n--- Scenario {i+1}: {scenario['name']} ---")
        print(f"  Raw input: {scenario['inputs']} -> normalized: [{', '.join(f'{v:.3f}' for v in normalized)}]")
        print(f"  Output: {outputs[0]:.4f} -> {decision}")

    # -----------------------------------------------------------------------
    # Sweep analysis: How distance affects behavior at different health levels
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SWEEP: How distance affects combat decisions (AK47 vs AK47, normal situation)")
    print("=" * 72)
    print(f"{'Distance':>10} {'Health':>8} {'Duck':>8} {'Jump':>8} {'Hide':>8} {'Move':>8} {'Strafe':>8}")
    print("-" * 66)

    for health in [1.0, 0.0, -1.0]:
        for dist in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            inputs = [health, dist, 0.0, 0.0, 1.0, 0.0]
            outputs = combat_nn.propagate(inputs)
            health_label = {1.0: "high", 0.0: "mid", -1.0: "low"}[health]
            dist_label = {-1.0: "near", -0.5: "near-mid", 0.0: "mid", 0.5: "mid-far", 1.0: "far"}[dist]
            print(f"{dist_label:>10} {health_label:>8} "
                  f"{outputs[O_DUCK]:>8.3f} {outputs[O_JUMP]:>8.3f} "
                  f"{outputs[O_HIDE]:>8.3f} {outputs[O_MOVETYPE]:>8.3f} "
                  f"{outputs[O_STRAFE]:>8.3f}")
        print()

    # -----------------------------------------------------------------------
    # Export network data as JSON for external analysis
    # -----------------------------------------------------------------------
    net_data = {
        "combat_nn": {
            "layers": combat_nn.layers,
            "weights": combat_nn.weights,
            "biases": combat_nn.biases,
            "input_names": ["health", "distance", "enemy_weapon", "weapon", "ammo", "situation"],
            "output_names": ["duck", "jump", "hide", "movetype", "strafe"],
        },
        "collision_nn": {
            "layers": collision_nn.layers,
            "weights": collision_nn.weights,
            "biases": collision_nn.biases,
            "input_names": ["left_dist", "middle_dist", "right_dist"],
            "output_names": ["steer_direction"],
        },
        "weapon_ranges": WEAPON_RANGES,
    }

    with open("joebot_nn_data.json", "w") as f:
        json.dump(net_data, f, indent=2)
    print("\nNetwork data exported to joebot_nn_data.json")


if __name__ == "__main__":
    run_demo()
