# 03. Dataset and Training

## Turn assignment

The robot is hardcoded to always play O and always move second. Evidence:
`CompTurn()` in `play_TicTacToe.py` always assigns the robot's candidate move
as `board[i] = 1`, and `parse_board_state()` maps `red -> 1`, `blue -> -1`.
The robot's role is not a parameter, it is baked into the algorithm. Making
the robot able to play either color would require passing a `robot_player`
argument through `CompTurn`/`minimax`, plus matching training demonstrations
for placing X.

At runtime, `play()`'s main loop only waits for the human on iterations after
the first (`i != 0`). The very first loop iteration goes straight to camera
capture and a robot move. In practice, the human must physically place the
first X on the board before launching the script; the script does not pause
for a first human move on its own.

## Dataset structure

90 training episodes total: 9 target grid cells times 10 demonstrations per
cell. Each episode is a single pick-and-place: one Red/O tile picked up and
placed at one target cell. The dataset does not contain multi-move games,
only isolated single-placement demonstrations.

Recording used LeRobot's standard `python -m lerobot.record` module
(teleoperation via the leader arm). There is no committed project script
that drives the recording session itself. Exact command in
`07_command_reference.md`.

## Board-state planning for recording

`TicTacToe_with_SO101/src/lerobot/scripts/ticTacToe/board_generator.py`
decides what the 8 non-target cells should contain before each recorded
demonstration, so the policy sees varied visual context per target cell
rather than always demoing against the same board.

For a given target cell, it generates 10 configurations, one per
piece-count scenario:

| Scenario (X, O) | Pieces on board excluding target cell |
|---|---|
| (0, 0) | 0 (empty board) |
| (1, 0) | 1 |
| (1, 1) | 2 |
| (2, 1) | 3 |
| (2, 2) | 4 |
| (3, 2) | 5 |
| (3, 3) | 6 |
| (4, 3) | 7 |
| (4, 4) | 8 |

Piece positions among the remaining 8 cells are chosen with `random.sample`
(randomized placement, fixed count for each scenario), validated by
`is_valid_game_state()` against two constraints: turn-order legality (X count
equals O count, or O count plus 1, since X always moves first) and no
pre-existing winning line on the board.

Concretely: recording a demo for a given target cell does not mean the other
8 cells are empty. Only the (0,0) scenario's single demo has a fully empty
board. The other 9 demos place up to 8 pieces randomly across the remaining
cells.

Known gap in `board_generator.py` (marked as a TODO in the source, near line
100): if a scenario cannot find a valid configuration within 1000 attempts,
it is silently skipped instead of retried. This can under-fill certain
scenarios for certain target cells, most likely for scenarios with higher
piece counts near diagonal-heavy positions, where the no-winning-line
constraint is harder to satisfy.

## Physical tokens used in recording

Wooden tile pieces slotted into a fixed 3x3 frame, matching the color
scheme read by the Gemini vision prompt at inference time (`Blue` = X,
`Red` = O).

## Policy: Action Chunking Transformer (ACT)

ACT is an imitation-learning policy that predicts short chunks of future
actions from current joint state and camera images, using a conditional
variational autoencoder during training (encoder discarded at inference,
latent set to zero).

By default, ACT is single-task. This project needed one policy that
generalizes across 9 different target cells. The fix: condition the policy
on a task-instruction embedding (a string like `"Place at Position N"`),
letting a single trained policy act differently depending on which cell it's
told to target. This is a modification to LeRobot's ACT implementation
(`TicTacToe_with_SO101/src/lerobot/policies/act/`), not stock behavior.

## Training run parameters

- Hardware: RTX 5090.
- 25,000 steps, 40 epochs, approximately 160 minutes wall time.
- Batch size 64 (increased from ACT's default, which reduced the number of
  steps needed for convergence).
- Otherwise default ACT hyperparameters as shipped by LeRobot.
- A checkpoint trained to 15,000 steps performed comparably; 25,000 was not
  strictly necessary.

Training used LeRobot's standard `python -m lerobot.scripts.train` module
against the recorded dataset, unmodified aside from the ACT policy change
described above. Exact command and full `ACTLangConfig` field reference in
`06_act_configuration.md` and `07_command_reference.md`.

## Why the policy barely uses vision

Attention-interpretability analysis (using Ville Kuosmanen's Physical AI
Interpretability Toolkit, integrated into this fork) found the trained
policy attends almost entirely to task instruction and joint state, and
barely to vision.

This follows directly from the dataset design: the robot's token color
(Red/O) never varies across any of the 90 episodes. Only cell position and
background clutter vary. Since color carries no information relevant to the
task, the model has no incentive to look at the image to distinguish
anything. A future dataset that trains the robot to place either color would
make color task-relevant, which should increase reliance on vision.

Result videos referenced in the original write-up are not stored in this
repository (see `05_known_issues.md` for the state of
`output/attention_analysis_results/`).

## Known limitation: this dataset is not environment-agnostic

The 90-episode dataset (9 cells x 10 demos, `board_generator.py`'s scenarios)
varies board occupancy, which cell has a piece and where, but does not vary
anything about the physical recording environment itself: camera position,
lighting, table background, or board placement are held fixed across every
episode.

This is a deliberate tradeoff, not an oversight. Varying board occupancy
stresses the model's task-conditioning pathway (learning to distinguish
"place at cell 5" from "place at cell 3"), but does not stress its visual
grounding. Since the target-cell signal is carried entirely by the task
instruction token (see `06_act_configuration.md`), and the camera view of
each cell's physical location never changes, the model has no training
pressure to use vision to locate the cell. It can succeed by learning a
near-fixed joint trajectory per task token instead, which is consistent with
the low vision-attention finding above.

Practical consequence: this policy is expected to work reliably only with
the camera, table, and board in the same physical position and lighting
used during recording. Moving the camera, changing lighting, or relocating
the board is likely to degrade performance, since the model was never
required to compensate for such changes visually. See
`01_hardware_setup.md` for how tightly the camera pipeline is calibrated to
one fixed setup.

A dataset that produced a genuinely environment-agnostic policy would need
to vary the nuisance factors (camera pose, lighting, board position,
background) across episodes while keeping the task-relevant signal (where
each cell actually is, relative to the camera) the thing the model has to
learn to track visually. That is a larger, separate recording effort and is
out of scope for the current 90-episode dataset.
