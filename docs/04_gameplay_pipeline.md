# 04. Gameplay Pipeline

Entry point: `python -m lerobot.play_TicTacToe` (see `02_software_setup.md`
for full command with required arguments).

Source: `TicTacToe_with_SO101/src/lerobot/play_TicTacToe.py`.

This project runs two independent camera-consumption paths; see
`01_hardware_setup.md`, "Camera rig," for the full breakdown. Steps 1-3
below cover only the Gemini board-reading path, which uses one hardcoded
camera (`camera_index = 2`). The ACT policy's own observation, consumed
inside `call_policy()` at step 7, uses whatever cameras are configured in
`--robot.cameras=` (both the front-angled and top-down cameras for this
project), via `robot.get_observation()`, entirely separate from the Gemini
capture.

## Per-turn sequence

1. **Capture (Gemini path only).** `get_grid_image()` reads a frame from the
   fixed camera used for board-reading (`camera_index = 2`).
2. **Rectify.** `crop_image()` crops to the board region using fixed
   percentage offsets, `transform_to_top_view()` perspective-warps 4
   hardcoded corner points to a clean 400x400 top-down view, then a final
   crop and 180-degree rotation produce the frame used for vision. All of
   these values are calibrated to one specific physical camera and board
   position; see `01_hardware_setup.md`.
3. **Perceive.** `get_LLM_output()` sends the rectified image to Gemini
   (`gemini-2.0-flash`) with a fixed prompt asking for the state of each of
   the 9 cells (`Empty`/`Red`/`Blue`). This is the only role Gemini plays in
   the pipeline: reading pixels into structured text. It is not told which
   color is "its own," is not asked to choose a move, and has no notion of
   whose turn it is; every downstream decision (win check, move selection)
   happens in plain deterministic code, not in the model.
4. **Parse.** `parse_board_state()` converts Gemini's text response into a
   9-element vector: `-1` for Blue/X, `1` for Red/O, `0` for empty.
5. **Check for game over.** `analyzeboard()` runs against whatever board
   Gemini just reported, every single loop iteration, before the robot
   considers a move. It checks all 8 winning lines (3 rows, 3 columns, 2
   diagonals) for three-in-a-row and returns the winning player's value
   (`1` for O/robot, `-1` for X/human) or `0` if there is no winner yet. If
   nonzero, or the board is full, `play()` announces `"Game Over"` and
   breaks out of the loop; the robot does not move that turn. `play()` does
   not currently distinguish a robot win, a human win, or a draw; all three
   take the same branch (see `05_known_issues.md`).
6. **Decide.** If the game is not over, `CompTurn()` runs `minimax()`
   locally, a plain game-tree search with no learned model, to find the
   robot's optimal empty cell.
7. **Act.** The chosen cell becomes an instruction string,
   `"Place at Position N"`. `call_policy()` hands this to the trained ACT
   policy. Inside `call_policy()`:
   - The ACT checkpoint is loaded fresh via `make_policy()` on every call
     (see `05_known_issues.md`, no caching across turns).
   - A control loop runs at `fps` Hz for up to `robot_turn_time_s` seconds
     (default 30s), or until a keyboard early-exit is triggered.
   - Each tick: read a fresh observation (`robot.get_observation()`) of
     every configured camera plus joint state (`SO101Follower.get_observation()`
     iterates all cameras in `config.cameras`, both the front-angled and
     top-down feeds for this project), run one policy forward pass via
     `predict_action()` (which internally manages ACT's action-chunk queue,
     only re-running the full transformer when the queue empties), send the
     resulting single-step joint action to the robot.
   - The turn ends purely on the timer or manual exit, not on any
     detection that the placement succeeded. Whether it actually worked is
     only revealed by the next camera read and Gemini call, at the top of
     the following loop iteration (see `05_known_issues.md`).
8. **Announce.** `log_say()` (text-to-speech) announces turn changes and
   game-over state, gated by `cfg.play_sounds`.

The loop then waits `player_turn_time_s` seconds for the human's physical
move before repeating from step 1, except on the very first iteration, which
skips the human-wait (see `03_dataset_and_training.md`, "Turn assignment").

## Configuration

`TicTacToeConfig` (dataclass in `play_TicTacToe.py`) exposes:

- `robot`: robot connection config (type, port, id).
- `policy`: path to the trained ACT checkpoint.
- `play_sounds`: enable/disable text-to-speech.
- `fps`: control loop rate during policy execution.
- `robot_turn_time_s`, `player_turn_time_s`: per-turn time budgets.
- `use_videos`: whether dataset-style feature encoding uses video (relevant
  to the mock metadata built for `make_policy`, not to dataset recording
  itself).

## Related files

- `TicTacToe_with_SO101/src/lerobot/scripts/ticTacToe/board_generator.py`:
  used during dataset planning, not at runtime. See `03_dataset_and_training.md`.
- `TicTacToe_with_SO101/src/lerobot/scripts/ticTacToe/image_transformation_testing.py`:
  standalone script for testing/recalibrating the camera crop and
  perspective-warp pipeline, without running the full game loop.
- `TicTacToe_with_SO101/src/lerobot/record.py`,
  `TicTacToe_with_SO101/src/lerobot/scripts/train.py`: stock upstream
  LeRobot CLI entry points used for data collection and training.
