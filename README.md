# Evolution

Evolution trains genetic programs or CPU convolutional neural networks to play a fast,
local Space Invaders approximation. Both agent types evolve without backpropagation.
The default executable needs no Atari ROM, ALE, CUDA, or CoreNN. Training runs headless;
an optional SDL2 viewer uses exactly the same simulator and agent interpreter.

## Build

Requirements: CMake 3.16 or newer, a C++11 compiler, threads, and Lua 5.4 development
files. SDL2 development files enable the viewer. On this laptop they are already installed.
The installed oneDNN v3 CPU runtime enables batched CNN matrix multiplication.
It uses OpenMP (or a sequential oneDNN build); development headers are optional.
CMake reports the selected CNN backend. If oneDNN/OpenMP is missing, it builds the
portable scalar fallback. To select that fallback explicitly, use `-DEVOLUTION_USE_DNNL=OFF`.

```sh
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release
cmake --build Build --parallel 4
ctest --test-dir Build --output-on-failure
```

The executable is `Bin/Evolution`. Run the following commands from the project root;
local simulation does not depend on the working directory or any ROM assets.

For a build without SDL2, configure with `-DEVOLUTION_BUILD_VIEWER=OFF`.
If SDL2 cannot be found, CMake builds headless training and reports that the viewer is disabled.
Use `-DEVOLUTION_RUNTIME_DIR=/absolute/output/path` to keep an alternate build's
executables separate from `Bin`.

## Train and watch

```sh
# Headless training; automatically starts from invaders_best if it exists.
Bin/Evolution train --generations 100 --population 128 --save invaders_best

# Watch the saved agent.
Bin/Evolution watch --load invaders_best

# Continue training from the previous best (same command on subsequent runs).
Bin/Evolution train --generations 100

# Keep training until Ctrl+C, saving each new best.
Bin/Evolution train --indefinite

# Explicitly start over, replacing the save checkpoint with the new run's best.
Bin/Evolution train --fresh --save invaders_new

# Seed a run from a different checkpoint.
Bin/Evolution train --load invaders_best --save invaders_variant

# Watch the built-in demonstration pilot, or play yourself.
Bin/Evolution watch
Bin/Evolution play

# Measure simulation, observation packing, and demo-policy throughput.
Bin/Evolution benchmark --frames 10000000
```

The viewer supports left/right arrows or A/D to move, Space to fire, P to pause,
N to advance one step while paused, R to restart, and Escape to quit.
The score is displayed inside the game at the top left, with lives at the top right.
Score, lives, wave, and frame count also appear in the window title. `--fps` changes
playback speed, not simulation rules. The viewer exits after `--episodes` games
(one by default); the final score is also printed to the console.

Training options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--agent` | `program`, or detected from a checkpoint | `program` or `cnn` |
| `--generations` | 100 | Number of generations to evaluate |
| `--indefinite` | Off | Run until Ctrl+C; cannot combine with `--generations` |
| `--population` | 128 | GA candidates, or ES perturbations (must be even for ES) |
| `--elite` | population / 16, at least 1 | GA only: best candidates retained unchanged |
| `--optimizer` | `ga` | `ga` for selection/mutation; `es` for CNN evolution strategies |
| `--threads` | Available logical CPUs | Evaluation workers, capped at population size |
| `--batch-size` | 4 | CNN training only: maximum concurrent games per worker; 1 evaluates one game at a time |
| `--episodes` | 3 | Seeds evaluated per agent; scores are averaged |
| `--seed` | 123 | Evolution seed and first simulation seed |
| `--max-frames` | 3600 | Maximum simulation ticks per episode |
| `--save` | `invaders_best` / `invaders_cnn_best` | Best-so-far checkpoint, replaced atomically; separate defaults per agent type |
| `--load` | Save path, if it exists | Explicit checkpoint to seed the population |
| `--fresh` | Off | Start from scratch instead of loading the save checkpoint; cannot combine with `--load` |
| `--mutation-rate` | 0.1 | CNN GA only: probability of perturbing each parameter, in (0, 1] |
| `--mutation-sigma` | 0.05 | CNN GA only: Gaussian perturbation standard deviation, in (0, 8] |
| `--es-sigma` | 0.02 | CNN ES only: Gaussian perturbation standard deviation, in (0, 8] |
| `--learning-rate` | 0.01 | CNN ES only: Adam step size, in (0, 1] |
| `--weight-decay` | 0.005 | CNN ES only: L2 coefficient, in [0, 1]; zero disables decay |

Parallel evaluation is enabled by default: this laptop uses 8 workers. Workers stay
alive across generations and draw agents from a shared queue, so a worker that
finishes a shorter game can immediately take another agent. Set `--threads 4` to
limit CPU use, or `--threads 1` for serial evaluation. For example, to continue the
checkpoint in `Bin` with all 8 workers from the project root:

```sh
Bin/Evolution train --threads 8 --generations 1000 --save Bin/invaders_best
```

Training prints progress every 10 generations and at the final generation.
It announces which checkpoint it loads. The saved champion is kept unchanged
in the initial population; the other agents start as mutated copies. After each
completed generation, an improved best is saved immediately and announced with
`Saved new best`. The file is replaced atomically, so stopping training leaves the
last completed save available. In indefinite mode, Ctrl+C stops ongoing evaluations
promptly, saves any improvement among agents that completed all episodes, and exits.
Partial evaluations are discarded so their scores cannot be compared with complete
ones. An existing checkpoint is kept if its agent has not yet been re-evaluated;
if a fresh run is interrupted before any agent finishes, there is no new checkpoint
to save. An automatically loaded champion is re-evaluated
under the requested seeds and frame limit; a tie leaves the existing file untouched.
An unreadable or invalid checkpoint causes an error instead of silently starting over.
Paths are relative to your working directory, so use the same directory or an explicit
`--save` path when continuing a run.

Every agent sees the same sequence of episode seeds: `seed`, `seed + 1`, etc.
CNN actions are sampled from softmax probabilities. Each game has a separate action
RNG reset to its episode seed, so a fixed `--seed` replays the stochastic policy.
Candidates share these seeded action draws for comparable fitness evaluations.
Mutation and seeded tie-breaking run on the main thread, and each worker owns its own simulator and RNG.
Identical settings produce identical fitness and checkpoints regardless of worker count
or CNN batch size (with the same build and CPU math backend). There is no sleep, drawing, or allocation in the
simulation step. Changing the training seeds changes the task; try additional seeds
when judging an agent's generalization. To reproduce an evaluation in the viewer,
pass its `--seed` and `--max-frames` values along with `--load`.

The program agent reuses `Genotype`, `Gene`, and the bounded `Execution` interpreter.
Both agent types use the GA trainer by default and share a persistent worker pool.
It evaluates one shared population in parallel, keeps the best elites, and mutates
copies of elite parents. Equal-fitness candidates are shuffled with a seeded RNG
before selection, allowing neutral offspring to replace parents and accumulate
changes across plateaus, including with `--elite 1`. Better fitness always ranks
first; the saved checkpoint is only replaced on a strict improvement.
It saves the best score seen during the run. A checkpoint
contains the agent, not the full population or random-generator state. Continuing
training seeds a new population from that best agent rather than restoring the exact run.

## CPU convolutional agent

```sh
# Headless CNN training; resumes invaders_cnn_best when it exists.
Bin/Evolution train --agent cnn --threads 8 --generations 1000

# Run continuously; Ctrl+C stops safely after saving the best completed agent.
Bin/Evolution train --agent cnn --indefinite --threads 8

# Batch four independent CNNs per worker (the default).
Bin/Evolution train --agent cnn --threads 8 --batch-size 4 --generations 1000

# The viewer detects the CNN checkpoint automatically.
Bin/Evolution watch --load invaders_cnn_best

# Change the evolutionary mutation settings or start a separate experiment.
Bin/Evolution train --agent cnn --fresh --save cnn_experiment \
    --mutation-rate 0.1 --mutation-sigma 0.05

# Include visual observation and neural inference in the CPU benchmark.
Bin/Evolution benchmark --load invaders_cnn_best --frames 100000
```

The fixed architecture has **13,598 evolved parameters** (weights and biases):

| Stage | Output | Parameters |
| --- | --- | ---: |
| 8x8 max pooling of the 256x240 grayscale playfield | 32x30x1 | 0 |
| 3x3 valid convolution, stride 2, four filters, leaky ReLU | 15x14x4 | 40 |
| Flatten, dense layer, leaky ReLU | 16 | 13,456 |
| Linear logits, softmax sampling (temperature 1) | 6 actions | 102 |

The input contains stable grayscale silhouettes of aliens, the player, projectiles,
shields, and the saucer. Maximum pooling retains small projectiles. The image uses
intensities 1 for the player and its shot, 0.75 for aliens and the saucer, 0.5 for
enemy shots, and 0.25/0.5 for damaged/intact shield cells, over a black background.
Score text, lives indicators, animation, and simulation RAM are excluded. Rectangles
are max-composited directly into pooled cells, exactly as pooling those silhouettes
after full-resolution rasterization would do. Training requires neither a window
nor a full-resolution framebuffer. The viewer shows the same simulated game using
its regular colored sprites.

Training advances independent games together within each worker. At each tick,
their images and **distinct evolved weights** form a batch. Convolution is an
im2col batched matmul, followed by batched dense and output matmuls using the
[oneDNN CPU library](https://github.com/oneapi-src/oneDNN). An agent that finishes
all its episode seeds is replaced with another agent from the shared queue.
Only complete evaluations enter selection or checkpoint comparisons.

`--batch-size` sets the maximum games per worker, automatically reduced for small
populations to keep work available for the requested threads. Four is the default
because larger batches can exceed this laptop's CPU cache and slow training down.
`--batch-size 1` retains the same math backend while evaluating one game at a time.
Each worker limits oneDNN to one math thread, so `--threads` controls CPU parallelism.
Input, activation, and weight buffers and compiled matmul primitives are reused
during evaluation. Inference runs on the CPU; training uses no backpropagation. The network
samples an action every simulation tick from `exp(logit - max(logits))`, normalized
across the six actions. This permits exploration instead of always choosing the
largest logit, and subtracting the maximum avoids exponential overflow.
Training, benchmarks, and the viewer all use this policy. Hidden activations use leaky ReLU with
negative slope 0.01. Initial layer weights are sampled uniformly with bounds
`sqrt(6 / fan-in)` and biases start at zero. With the default `--optimizer ga`, each offspring copies an elite and
adds Gaussian noise to a random subset of **all layers' weights and biases**.
At least one parameter is selected per mutation; values are bounded to [-8, 8].
Elites, fitness, neutral tie-breaking, progress logging, and checkpoint saving use
the same rules as program-agent training. Mutation settings affect new offspring
when resuming; the saved agent's weights are loaded exactly.
The viewer uses the same inference backend with a single-game batch. Existing
CNN checkpoint files remain compatible, but previously saved agents now sample
actions instead of choosing the largest logit. Switching between the scalar and
oneDNN backends can change floating-point rounding and therefore sampled actions;
reproducibility across different backends is not guaranteed.

CNN checkpoints are versioned, validate the architecture and every parameter,
and preserve float values exactly on round trip. Agent type is detected from
`--load` or an existing `--save`; explicitly selecting the wrong type is an error.
Without a checkpoint or explicit `--agent cnn`, training defaults to the program
agent. `watch --agent cnn` loads the default CNN checkpoint, while
`benchmark --agent cnn` without `--load` measures a seeded random CNN.
CNNs have a separate default save filename so existing program checkpoints remain
available. Paths are still relative to the working directory.

### CNN evolution strategies

Use `--optimizer es` for the mirrored-perturbation update from
[Salimans et al., 2017](https://arxiv.org/abs/1703.03864), with centered ranks and Adam:

```sh
# Start ES from the existing champion and save to a separate checkpoint.
Bin/Evolution train --agent cnn --optimizer es \
    --load Bin/invaders_cnn_best --save Bin/invaders_cnn_es_best \
    --population 128 --generations 100 --threads 8

# Continue from the saved ES policy until Ctrl+C.
Bin/Evolution train --agent cnn --optimizer es --indefinite \
    --save Bin/invaders_cnn_es_best --threads 8

# Fresh ES run with explicit optimizer settings.
Bin/Evolution train --agent cnn --optimizer es --fresh --save cnn_es_experiment \
    --es-sigma 0.02 --learning-rate 0.01 --weight-decay 0.005
```

For an even population `N`, each generation draws `N/2` independent Gaussian
directions `epsilon` over **every weight and bias**. It evaluates both
`theta + sigma * epsilon` and `theta - sigma * epsilon` on the same episode and
action RNG seeds. Returns are ranked across all `N` perturbations and mapped to
`[-0.5, 0.5]`. Equal returns receive their average rank, so equal mirrored scores
cancel and a completely tied population supplies zero reward gradient.

The update direction is:

```text
g = sum((rank_plus[i] - rank_minus[i]) * epsilon[i]) / (N * sigma)
```

Adam performs ascent using `g - weight_decay * theta`, with beta1=0.9,
beta2=0.999, epsilon=1e-8, and bias correction. L2 applies to biases as well as
weights. This uses the paper's explicit `1/sigma` normalization; the released
[OpenAI starter implementation](https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
omits that fixed scale factor. Consequently its regularization settings need not
be equivalent here. The optimizer estimates a direction from complete game
returns and uses the existing CNN architecture and softmax action sampling.

`--generations` counts central-policy updates. An initial central evaluation
establishes the baseline, then each generation evaluates `N` perturbations and
one updated center, each across `--episodes` seeds. Only fully evaluated central
policies can replace the best checkpoint, and only on a strict score increase.
The current center continues updating even when its score falls below the saved
best. Logs report central score, best central score, perturbation mean/best,
reward-gradient norm, and applied step norm. A zero reward gradient may still
produce a step through L2 regularization or Adam's existing momentum.

Central weights are projected to the checkpoint range `[-8, 8]` after each update.
Temporary perturbations may exceed that range so the Gaussian pairs stay
symmetric; they are never saved. `--elite` and `--mutation-*` are rejected in ES
mode. Evaluation uses the same batching and worker pool as GA, and fixed settings
give identical results across worker counts and batch sizes on the same backend.

Ctrl+C in indefinite mode skips an incomplete perturbation batch and keeps the
best completed central evaluation. An incomplete center is never saved. The
checkpoint format remains policy-only: loading a GA or ES checkpoint starts a
new center from those exact weights, with fresh Adam moments and RNG state.
It does not resume the precise optimizer trajectory. Evaluation seeds remain
`seed`, `seed + 1`, etc.; assess generalization with additional seeds.

## Simulation model

The arena is 256 by 240 integer units. Each action advances one nominal 60 Hz tick.
A 5 by 11 formation moves sideways, descends at the edges, and speeds up as enemies
die. The player has three lives, movement and combined movement/fire actions,
one shot in flight, a firing cooldown, and brief protection after losing a life.
Bottommost enemies shoot downward. Four destructible bunkers stop bullets from
either side, and a bonus saucer crosses the top. Clearing a wave starts a faster
one. Losing all lives, an enemy landing, or the frame limit ends the episode.

Kills award 30 points for the top row, 20 for the next two, and 10 for the bottom
two. The saucer and completing a wave each award 100. Fitness is the sum of these
rewards; survival alone earns no points. Collisions use swept vertical bullet
segments so fast projectiles cannot skip targets.

This is an intentionally simplified game, not Atari emulation: timing, physics,
enemy firing, shields, and observations differ. Existing ALE agents can use the
same checkpoint syntax, but their learned RAM interpretation does not transfer.
Train new agents for this simulator. The CNN's pooled visual input is produced
headlessly; full-resolution display rendering is optional.

## Program agent interface

`SpaceInvaders` implements the existing `Game` interface. `do_action()` returns the
score gained that tick, `getState()` returns a stable 128-byte buffer, and `restart()`
restores the initial state and random seed. `reset(seed)` selects a new deterministic
episode. A terminal game returns zero reward and does not advance until reset.
`world()` exposes a read-only snapshot for the renderer.

Actions are `0 = idle`, `1 = fire`, `2 = left`, `3 = right`, `4 = left + fire`,
and `5 = right + fire`. The interpreter's first register selects the action modulo six.
Eight registers are cleared before every decision, matching the existing evaluator.
Integer additions saturate instead of overflowing.
`CALL` targets a gene by index; targets beyond the gene count clamp to the last
gene. Calls share registers and the execution budget, and return to the caller.

Observation layout (coordinates use arena units):

| Bytes | Meaning |
| --- | --- |
| 0–1 | Player x, lives |
| 2–4 | Player shot active, x, y (inactive coordinates are zero) |
| 5–7 | Formation x clamped at zero, formation y, direction (0 left, 1 right) |
| 8–11 | Remaining aliens, wave (capped at 255), fire cooldown, terminal flag |
| 12–13 | X and y of the lowest active enemy bullet; zero if none |
| 14–15 | Score modulo 65536, low byte first |
| 16–70 | Alive flags for each invader, row-major, top to bottom and left to right |
| 71–94 | Eight enemy bullets, each encoded as active, x, y |
| 95–112 | Shield health, four 2-bit cells per byte, lowest bits first |
| 113–114 | Frame count modulo 65536, low byte first |
| 115–118 | Saucer active, x, direction (0 left, 1 right), player protection ticks |
| 119–120 | Signed formation x + 256, low byte first (origin can move left of zero) |
| 121–127 | Reserved, zero |

Shield cells are ordered by bunker from left to right, then by row and column
within each 3 by 6 bunker. Health is 0, 1, or 2. Observation packing is lazy and
has no effect on simulation or random-number generation.

## Validation and legacy ALE

Tests cover replay/reset determinism, collisions, shields, lives, invasion, waves,
frame limits, interpreter overflow and function calls, neutral parent selection,
checkpoint validation and resume, identical training across worker counts, and
agreement between viewer scores and training.
CNN checks cover pixel pooling against full-resolution rasterization, analytical
forward-pass results, softmax action frequencies and numerical stability, seeded
action replay, mutation of every layer, float checkpoint round trips,
malformed checkpoint rejection, and parallel/headless/viewer parity. Batched tests
compare different networks against independent double-precision reference math,
check exact singleton/batch logits, and verify identical fitness and checkpoints
across batch sizes, worker counts, partial batches, and game-slot reuse.
ES checks cover tie-aware centered ranks, analytical gradient sign and scaling,
mirrored noise including biases and boundary weights, independent Adam calculations,
optimization of a known objective, worker/batch determinism, checkpoint loading,
central-policy saves, interruption, and argument validation.
SDL tests use its dummy video driver, so they also run without a desktop session.

The old ROM trainer remains available as a separate optional target:

```sh
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release -DEVOLUTION_BUILD_ALE=ON
cmake --build Build --target EvolutionALE --parallel 4
(cd Bin && ./EvolutionALE)
```

It needs ALE 0.6.0 sources (`ALE_SOURCE_DIR`), zlib, and the original ROM/checkpoint
files under `Bin`. The local executable never links to ALE. The unfinished
`NetTraining.cpp` CoreNN/CUDA experiment is excluded from both executables.
