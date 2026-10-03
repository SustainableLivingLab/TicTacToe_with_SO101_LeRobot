# 05. Known Issues

## Fixed

- `PreTrainedConfig.from_pretrained()`
  (`TicTacToe_with_SO101/src/lerobot/configs/policies.py`) wrote the
  downloaded checkpoint's config to a `tempfile.NamedTemporaryFile`, then
  tried to reopen that same path with a plain `open()` while the original
  file handle was still held open. Windows denies this (POSIX allows
  reopening an already-open file; Windows does not), so loading any
  checkpoint via `--policy.path=` on Windows crashed with
  `PermissionError: [Errno 13] Permission denied: '<temp file path>'`,
  before the policy even finished loading. This affects every inference
  command (`play_TicTacToe.py`) and any script calling `make_policy()` with
  a Hub or local checkpoint path on Windows. Fixed by creating the temp
  file with `delete=False` and explicitly closing and deleting it after use
  instead of relying on the `with` block's own (too-early) cleanup.
- A local inference/training venv built from `pip install -e .` on a
  machine with an actual NVIDIA GPU can still end up with a CPU-only
  `torch` build (`torch==X.Y.Z+cpu`), if `torch` happened to resolve from a
  cached or previously-installed wheel instead of a CUDA build.
  `torch.cuda.is_available()` returns `False` and `play_TicTacToe.py`/any
  policy-loading script silently falls back to CPU (`WARNING:root:No
  accelerated backend detected`), which is correct behavior but much
  slower than necessary when a GPU is actually present. Check with
  `python -c "import torch; print(torch.cuda.is_available())"` once after
  any environment setup; if `False` on a GPU machine, reinstall explicitly
  from the matching CUDA wheel index, e.g.
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps`
  (match the index's CUDA version to what `nvidia-smi` reports support
  for; `torchvision` must be reinstalled from the same index too, since a
  CPU/CUDA `torch` and `torchvision` mismatch fails at import time with
  `RuntimeError: operator torchvision::nms does not exist`).
- `decode_video_frames_torchvision()`
  (`TicTacToe_with_SO101/src/lerobot/datasets/video_utils.py`) called
  `torchvision.io.VideoReader`, which was removed entirely in newer
  torchvision releases (confirmed on `torchvision==0.29.0`: `pyav` is the
  only viable decode backend on Windows in this fork, since `torchcodec` is
  conditionally excluded on `sys_platform == 'win32'` in `pyproject.toml`,
  and the `video_reader` backend never worked without a from-source
  torchvision build). Every dataset load that actually reads a video frame,
  training included, crashed with `AttributeError: module 'torchvision.io'
  has no attribute 'VideoReader'`. Rewritten to call PyAV (`av`) directly
  instead of through torchvision's removed wrapper, producing the same
  uint8, channel-first frame tensors the rest of the pipeline expects.
  Found and fixed while testing the dataset format re-export described in
  `03_dataset_and_training.md`.
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
