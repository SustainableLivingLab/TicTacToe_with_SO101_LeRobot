# 05. Known Issues

## Fixed

- Duplicate import of `hw_to_dataset_features` and `make_robot_from_config`
  in `play_TicTacToe.py`. Removed the redundant `lerobot.record` import;
  both are now imported once each.
- Unguarded regex match in `call_policy()`. `matches[-1]` on
  `re.findall(r'Place at position \d+', instruction, re.IGNORECASE)` now
  raises a clear `ValueError` instead of an opaque `IndexError` if the
  instruction string does not contain that pattern.
- `TicTacToe_with_SO101/src/lerobot/scripts/ticTacToe/TicTacToeAlgorithm.py`,
  a stale duplicate of the `analyzeboard`/`minimax`/`CompTurn`/`print_board`
  logic in `play_TicTacToe.py`, has been deleted.
- Missing null-check on `get_grid_image()`'s return value. `play()` now
  checks for a failed camera read and retries the loop instead of crashing
  in `crop_image`/`get_LLM_output`.
- Hardcoded Gemini API key. Moved to the `GEMINI_API_KEY` environment
  variable, read from a gitignored `.env` file (see `02_software_setup.md`).
- Hardcoded piece colors. `play_TicTacToe.py`, `image_transformation_testing.py`,
  and `board_generator.py` all read piece colors from the `X_COLOR` and
  `O_COLOR` environment variables now (default `Blue`/`Red`), instead of
  hardcoding color literals. Different physical boards (e.g. Black X / Red
  O) no longer need a code edit, only a `.env` change. See
  `02_software_setup.md`, "Piece colors."
- Second hardcoded Gemini API key found in
  `TicTacToe_with_SO101/src/lerobot/scripts/ticTacToe/image_transformation_testing.py`
  (a plaintext key, separate from the one already fixed in
  `play_TicTacToe.py`). Moved to the same `GEMINI_API_KEY` environment
  variable pattern.

## Design limitations

- The robot only ever plays O and only ever moves second; this is hardcoded,
  not configurable. See `03_dataset_and_training.md`.
- The camera crop, perspective-warp corner points, and camera index are all
  hardcoded to one specific physical setup. Moving the camera or board
  requires manually recalibrating these values. See `01_hardware_setup.md`.
- `board_generator.py` silently skips a scenario if it cannot find a valid
  board configuration within 1000 attempts, instead of retrying with relaxed
  constraints. See `03_dataset_and_training.md`.
- The dataset is not environment-agnostic. Camera position, lighting, table
  background, and board placement are fixed across all 90 episodes, so the
  trained policy is only expected to work reliably in that same physical
  setup. This is a deliberate scope decision for the current dataset, not a
  bug. See `03_dataset_and_training.md`, "Known limitation: this dataset is
  not environment-agnostic," for what a future dataset would need to change.

## Runtime behavior gaps

- **Win/draw/loss are not distinguished.** `analyzeboard()` returns which
  player's value completed a line (`1` for O/robot, `-1` for X/human) or `0`
  for no winner yet, but `play()` only branches on whether the result is
  zero or non-zero. A robot win, a human win, and a draw (no winning line,
  board full) all collapse into the same `"Game Over"` branch and the same
  spoken message. There is no code path that reports who actually won.
- **No completion detection during the robot's turn.** `call_policy()` runs
  the ACT policy in a fixed time loop (`robot_turn_time_s`, default 30
  seconds) with no check for whether the pick-and-place actually succeeded
  mid-motion. The turn simply ends when the timer runs out (or on manual
  keyboard exit), regardless of outcome. The only thing that reveals
  whether the placement worked is the next camera read and Gemini call, at
  the start of the following loop iteration, after the fact.
- **No presence check at the pickup spot.** Nothing in the pipeline verifies
  a Red/O tile is actually present at the fixed pickup location before the
  arm attempts to grab it. If the pickup spot is not refilled between
  turns, the arm will still attempt its trained pick motion against an
  empty spot, with no detection or fallback. This is an operating-procedure
  requirement (always refill promptly), not something the current policy or
  game loop guards against.
- **Policy checkpoint reloaded every turn.** `call_policy()` calls
  `make_policy(cfg.policy, ds_meta=cfg.metadata)` fresh on every invocation,
  once per robot turn, rather than loading the checkpoint once at script
  startup and reusing it. This adds avoidable load time to every single
  robot turn.
- **Only one physical trajectory is learned per grid cell.** Because the
  pickup spot is fixed and the board itself is fixed, the pick motion and
  each cell's place motion are each a single repeated joint-space
  trajectory across all 10 demos for that cell, not a range of trajectories
  to the same destination from different starting conditions. The dataset
  varies background clutter (see `03_dataset_and_training.md`) but not the
  physical pick location, arm starting pose, or place location. Given a
  fixed pickup spot is a deliberate design choice for this project's scope
  (not a flaw, since deployment also always uses a fixed spot), this means
  the trained policy is closer to 9 fixed motor scripts (one per cell) than
  a policy that generalizes joint-space reaching. Consistent with the
  near-zero vision attention already documented; see
  `03_dataset_and_training.md`, "Known limitation: this dataset is not
  environment-agnostic."

## Repository state

- `TicTacToe_with_SO101/output/attention_analysis_results/` and roughly 45
  files under `TicTacToe_with_SO101/tests/artifacts/` are tracked via Git
  LFS but have no real binary content in this checkout, only pointer stubs.
  GitHub rejects pushing broken LFS pointers, so these files are untracked
  from git (kept on disk locally, `.gitattributes` LFS rules remain in
  place). The `tests/artifacts/` files are upstream LeRobot test fixtures,
  unrelated to this project; the attention-analysis videos are project
  output that would need to be regenerated or re-sourced from wherever the
  original training run's results live.
