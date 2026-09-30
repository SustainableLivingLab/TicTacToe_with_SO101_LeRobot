# TicTacToe_with_SO101

A robot arm (SO-101) that plays Tic-Tac-Toe against a human, built on a
fork of [LeRobot](https://github.com/huggingface/lerobot).

Human plays X (Blue), always moves first. Robot plays O (Red), always
moves second, hardcoded.

## Pipeline

1. Camera captures the board, image is perspective-warped to a top-down view.
2. Gemini reads the state of each of the 9 cells (Empty/Red/Blue).
   Perception only, no move reasoning.
3. A local minimax search picks the robot's move. Plain game-tree search, not
   a learned model.
4. A trained, task-conditioned ACT policy executes the physical pick-and-place
   for the chosen cell.
5. Text-to-speech announces turns and game state.

Entry point: `python -m lerobot.play_TicTacToe`

## Documentation

Full documentation is in `docs/`:

- `docs/00_overview.md`: what SO-101 and LeRobot are, project objective,
  end-to-end pipeline.
- `docs/01_hardware_setup.md`: arm assembly, motor configuration,
  calibration, camera rig.
- `docs/02_software_setup.md`: installation, environment variables, running
  the project.
- `docs/03_dataset_and_training.md`: dataset design, data collection, ACT
  training parameters, interpretability findings.
- `docs/04_gameplay_pipeline.md`: the runtime game loop, file by file.
- `docs/05_known_issues.md`: known bugs, gaps, and cleanup items.
- `docs/06_act_configuration.md`: ACT model configuration, layer counts,
  and the task-instruction conditioning mechanism.
- `docs/07_command_reference.md`: every command from install to inference.
- `docs/08_dataset_90_boards.md`: all 90 board layouts for the recording
  session, one per demo.
- `docs/09_concept_overview_for_students.md`: the whole project explained
  in simple English, no code, for teaching or onboarding new students.

Read `docs/00_overview.md` first.

## Code contributions on top of upstream LeRobot

- Modified ACT to accept a task-instruction token as input, so one policy
  generalizes across all 9 grid positions instead of training 9 separate
  policies.
- Integrated [Ville Kuosmanen's interpretability
  toolkit](https://github.com/villekuosmanen/physical-AI-interpretability),
  extended to also visualize task-instruction attention.
- `TicTacToe_with_SO101/src/lerobot/play_TicTacToe.py`: full game loop.
- `TicTacToe_with_SO101/src/lerobot/scripts/ticTacToe/`: board-state planner
  for dataset recording and camera calibration/testing scripts.

## Built on LeRobot

This repo is a fork of Hugging Face's
[LeRobot](https://github.com/huggingface/lerobot), which provides the robot
drivers, dataset format, training/eval scripts, and the ACT policy
implementation this project builds on. For installing LeRobot itself, training
generic policies, or working with the `LeRobotDataset` format, see the
upstream [LeRobot docs](https://huggingface.co/docs/lerobot) and
`TicTacToe_with_SO101/docs/source/`.
