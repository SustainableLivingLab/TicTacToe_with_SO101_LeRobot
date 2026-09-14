# 04. Gameplay Pipeline

Entry point: `python -m lerobot.play_TicTacToe` (see `02_software_setup.md`
for full command with required arguments).

Source: `TicTacToe_with_SO101/src/lerobot/play_TicTacToe.py`.

## Per-turn sequence

1. **Capture.** `get_grid_image()` reads a frame from the fixed camera
   (`camera_index = 2`).
2. **Rectify.** `crop_image()` crops to the board region using fixed
   percentage offsets, `transform_to_top_view()` perspective-warps 4
   hardcoded corner points to a clean 400x400 top-down view, then a final
   crop and 180-degree rotation produce the frame used for vision. All of
   these values are calibrated to one specific physical camera and board
   position; see `01_hardware_setup.md`.
3. **Perceive.** `get_LLM_output()` sends the rectified image to Gemini
   (`gemini-2.0-flash`) with a fixed prompt asking for the state of each of
   the 9 cells (`Empty`/`Brown`/`Black`). This step is perception only, it
   does not choose a move.
4. **Parse.** `parse_board_state()` converts Gemini's text response into a
   9-element vector: `-1` for Black/X, `1` for Brown/O, `0` for empty.
5. **Decide.** `CompTurn()` runs `minimax()` locally, a plain game-tree
   search with no learned model, to find the robot's optimal cell.
   `analyzeboard()` checks for a completed win/draw before this step; if the
   game is already over, the loop ends.
6. **Act.** The chosen cell becomes an instruction string,
   `"Place at Position N"`. `call_policy()` hands this to the trained ACT
   policy via `predict_action()`, which drives the follower arm for up to
   `robot_turn_time_s` seconds to execute the pick-and-place.
7. **Announce.** `log_say()` (text-to-speech) announces turn changes and
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
- `TicTacToe_with_SO101/src/lerobot/scripts/ticTacToe/TicTacToeAlgorithm.py`:
  stale duplicate of the minimax logic now living in `play_TicTacToe.py`. See
  `05_known_issues.md`.
- `TicTacToe_with_SO101/src/lerobot/record.py`,
  `TicTacToe_with_SO101/src/lerobot/scripts/train.py`: stock upstream
  LeRobot CLI entry points used for data collection and training.
