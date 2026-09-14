# 05. Known Issues

## Code issues

- **Duplicate import** of `hw_to_dataset_features` in `play_TicTacToe.py`
  (once from `lerobot.record`, once from `lerobot.datasets.utils`). Second
  shadows the first. Harmless, but should be cleaned up.
- **Unguarded regex match** in `call_policy()`: `matches[-1]` on
  `re.findall(r'Place at position \d+', instruction, re.IGNORECASE)` raises
  `IndexError` if the instruction string does not contain that pattern.
- **`TicTacToe_with_SO101/src/lerobot/scripts/ticTacToe/TicTacToeAlgorithm.py`**
  is a stale duplicate of the `analyzeboard`/`minimax`/`CompTurn`/`print_board`
  logic now living in `play_TicTacToe.py`, including leftover debug
  `print()` calls inside `minimax`. Candidate for deletion.
- **No null-check** on `get_grid_image()`'s return value. If the camera read
  fails (`ret == False`), `image` is `None` and the next call
  (`crop_image`/`get_LLM_output`) crashes instead of failing gracefully.
- **Color/token naming inconsistency.** `play_TicTacToe.py` (runtime, Gemini
  prompt and parser) uses Black/Brown. `board_generator.py`'s `print_board`
  prints the same pieces as B/W ("White"), while internally still using X/O.
  Functionally consistent, the labels just differ between files. Standardize
  on Black/Brown in any new code, it matches the physical carrom coins and
  the Gemini prompt.

## Design limitations

- The robot only ever plays O and only ever moves second; this is hardcoded,
  not configurable. See `03_dataset_and_training.md`.
- The camera crop, perspective-warp corner points, and camera index are all
  hardcoded to one specific physical setup. Moving the camera or board
  requires manually recalibrating these values. See `01_hardware_setup.md`.
- `board_generator.py` silently skips a scenario if it cannot find a valid
  board configuration within 1000 attempts, instead of retrying with relaxed
  constraints. See `03_dataset_and_training.md`.

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
