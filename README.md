# TicTacToe_with_SO101

A robot arm (SO-101) that plays Tic-Tac-Toe against a human, built on top of the
[LeRobot](https://github.com/huggingface/lerobot) codebase.

## Project Summary

- Developed on top of the **LeRobot** codebase.
- Combines:
  - A **classical minimax solver** to pick the robot's next move.
  - **Gemini vision** to read the physical board state from a camera image.
  - An **Action Chunking Transformer (ACT)** policy to execute the pick-and-place.

Human always plays X (Black) and moves first; the robot always plays O (Brown)
and moves second. See `docs/FIELD_NOTES.md` for the full breakdown of why, and
every other build/data-collection detail.

## How it works

1. Camera captures the board, image is perspective-warped to a top-down view.
2. Gemini reads off the state of each of the 9 cells (Empty/Brown/Black).
   Perception only, no move reasoning.
3. A local minimax search picks the robot's best move. Plain game-tree search,
   not a learned model.
4. A trained, task-conditioned ACT policy executes the physical pick-and-place
   for the chosen cell.
5. Text-to-speech announces turns and game state.

Entry point: `python -m lerobot.play_TicTacToe`

## Technical details

- Trained a **task-conditioned ACT model** covering all 9 grid positions as a
  single policy (task instruction like `"Place at Position N"` conditions the
  model per-cell, instead of training 9 separate policies).
- Integrated [Ville Kuosmanen's interpretability
  toolkit](https://github.com/villekuosmanen/physical-AI-interpretability) to
  visualize attention across vision / joint-state / task-instruction inputs.

### Key observations

- **Joint state** receives ~100% attention in many cases. The model has
  learned a near-direct mapping from joint configuration to action.
- **Task instruction** consistently receives >50% attention. Task conditioning
  is what actually identifies the target cell.
- **Vision** receives very little attention. This tracks with the dataset design:
  the robot's token color never varies across training episodes, so vision
  carries no information the model needs. Full explanation in
  `docs/FIELD_NOTES.md` section 5.

## Code contributions on top of upstream LeRobot

- Modified ACT to accept a task-instruction token as input (multi-task
  conditioning across the 9 grid cells).
- Integrated Ville Kuosmanen's interpretability tool into the LeRobot codebase,
  extended to also visualize task-instruction attention.
- `TicTacToe_with_SO101/src/lerobot/play_TicTacToe.py`: full game loop (see
  "How it works" above).
- `TicTacToe_with_SO101/src/lerobot/scripts/ticTacToe/`: board-state planner
  for dataset recording and camera calibration/testing scripts.

**TODO:** task instruction is currently a required input to the policy; making
it optional is the next step.

## Known issues

See `docs/FIELD_NOTES.md` section 6 for the current list (hardcoded API key
still needs to move to an env var, a couple of missing guards, one stale
scratch file). Worth a read before extending this project.

## Next steps

- Train on more diverse and randomized board configurations. Vary table
  layout, lighting, and coin placement within cells, not just board occupancy,
  to encourage the policy to actually use vision.
- Generalize `CompTurn`/`minimax` to let the robot play either color.

## Field notes

`docs/FIELD_NOTES.md` has the detailed, code-verified notes on: turn order,
physical piece counts, the color/token naming inconsistency between scripts,
exactly how the 90-episode dataset is planned (`board_generator.py`'s
scenarios), why the policy ignores vision, and a full code map. Read that
before setting up a new recording session.

## Built on LeRobot

This repo is a fork of Hugging Face's
[LeRobot](https://github.com/huggingface/lerobot), which provides the robot
drivers, dataset format, training/eval scripts, and the ACT policy
implementation this project builds on. For installing LeRobot itself, training
generic policies, or working with the `LeRobotDataset` format, see the upstream
[LeRobot docs](https://huggingface.co/docs/lerobot) and
`TicTacToe_with_SO101/docs/source/`. None of that is duplicated here.
