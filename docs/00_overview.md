# 00. Overview

## Objective

Build a physical robot arm (SO-101) that plays Tic-Tac-Toe against a human
opponent on a real board with physical tokens, end to end: seeing the board
through a camera, deciding a move, and physically placing a token.

## What is LeRobot

LeRobot is Hugging Face's open-source library for real-world robotics in
PyTorch. It provides:

- Robot and teleoperator drivers (motor control, camera interfaces).
- A standard dataset format (`LeRobotDataset`) for recording and replaying
  robot demonstrations (camera frames, joint states, actions).
- Implementations of imitation-learning and reinforcement-learning policies,
  including ACT (Action Chunking Transformer), Diffusion Policy, and others.
- CLI tools for recording demonstrations (`python -m lerobot.record`),
  training policies (`python -m lerobot.scripts.train`), and evaluating them
  (`python -m lerobot.scripts.eval`).

This project is a fork of LeRobot version 0.2.0. The fork adds a Tic-Tac-Toe
specific game loop and dataset-planning script on top of the unmodified
LeRobot library, plus one modification to the ACT policy itself (see
`03_dataset_and_training.md`).

## What is SO-101

SO-101 is an open-source, 3D-printable robot arm designed by TheRobotStudio,
documented and supported directly in LeRobot. A full setup consists of two
arms:

- **Leader arm**: moved by hand during teleoperation. A human physically
  moves the leader arm, and the follower arm mirrors those movements. Used
  only during data collection, not during autonomous play.
- **Follower arm**: the arm that actually executes actions, either mirroring
  the leader during data collection or running a trained policy during
  autonomous operation. This is the arm that plays the game.

Each arm has 6 motors (STS3215 servos), one per joint: base/shoulder pan,
shoulder lift, elbow flex, wrist flex, wrist roll, and gripper. The two arms
use different gear ratios on some joints so the leader can be moved by hand
without much force while the follower has the torque to carry a real load.

Full assembly instructions live in `01_hardware_setup.md`, distilled from
LeRobot's own SO-101 documentation
(`TicTacToe_with_SO101/docs/source/so101.mdx`).

## How the game works, end to end

1. A human places the first token (X) on the physical board before starting
   the game script.
2. A camera captures an image of the board and a computer-vision pipeline
   warps it into a clean top-down view.
3. Gemini (a vision-language model) reads the image and reports which of the
   9 cells are empty, X, or O.
4. A local minimax search (plain game-tree search, not a trained model)
   computes the robot's optimal move given that board state.
5. A trained ACT policy, conditioned on the target cell, physically picks up
   an O token and places it in the chosen cell.
6. Steps 2 to 5 repeat until the board is won or full.

## Division of labor: who plays what

The human always plays X and always moves first. The robot always plays O
and always moves second. This is a hardcoded property of the current
implementation, not a rule enforced by the game itself. See
`04_gameplay_pipeline.md` for the exact code path and `03_dataset_and_training.md`
for why the training data reflects this.

## Document index

- `01_hardware_setup.md`: assembling and calibrating the SO-101 arms, camera
  rig setup.
- `02_software_setup.md`: installing dependencies, environment variables,
  running the project.
- `03_dataset_and_training.md`: how the training dataset was designed and
  collected, and how the ACT policy was trained.
- `04_gameplay_pipeline.md`: the runtime game loop, file by file.
- `05_known_issues.md`: known bugs, gaps, and cleanup items in the current
  code.
- `06_act_configuration.md`: ACT model configuration, every tunable field,
  and how the task-instruction conditioning mechanism works.
- `07_command_reference.md`: every command from install to inference, in
  order.
- `08_dataset_90_boards.md`: all 90 board layouts for the recording session,
  rendered as 3x3 grids, one per demo.
- `09_concept_overview_for_students.md`: the whole project explained in
  simple English, no code, for teaching or onboarding new students.
