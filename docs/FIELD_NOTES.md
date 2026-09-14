# Field Notes: TicTacToe_with_SO101

Working notes on how this project actually behaves, derived from reading the code
(not just the blog post). Written to be idiot-proof for whoever sets this up next,
including future us.

## 1. What plays what

- **Human = X (Black), always moves first.** Standard tic-tac-toe convention.
- **Robot = O (Brown), always moves second.** Hardcoded, not configurable.

Evidence: `CompTurn()` in `src/lerobot/play_TicTacToe.py` always assigns the robot's
candidate move as `board[i] = 1`, and `parse_board_state()` maps `brown -> 1`,
`black -> -1`. The robot's role is baked into the algorithm, not passed as a
parameter. If you ever want the robot to play X, `CompTurn`/`minimax` need a
`robot_player` argument instead of the hardcoded `1`, and the dataset would need
matching demos.

**Caveat on turn order at runtime:** `play()`'s main loop only waits for the human
(`busy_wait(cfg.player_turn_time_s)`) when `i != 0`, meaning it skips the human-wait
on the very first iteration and goes straight to camera capture plus robot move. In
practice this means **the human must physically place the first X on the board
before launching the script**; the script itself does not pause for a first human
move.

## 2. Physical pieces

- Standard retail tic-tac-toe sets ship **5 X + 5 O = 10 pieces** (symmetric, so
  the set works no matter who goes first).
- This project never needs the 5th O. Max board occupancy used anywhere in the
  code/dataset-planning is 4 X + 4 O (see `board_generator.py`'s `(4,4)` scenario,
  one move before a full board). The 5th O tile is simply unused spare.
- Full theoretical max for a real (possibly drawn) game is 5 X + 4 O = 9 on the
  board, never hit in the current dataset design since the target cell is
  always left empty for the demo.

## 3. Color/token vocabulary: watch for drift

Two different naming schemes are used across the codebase for the exact same two
colors:

| Context | X-equivalent | O-equivalent |
|---|---|---|
| `play_TicTacToe.py` (runtime, Gemini prompt + parser) | **Black** | **Brown** |
| `board_generator.py` (dataset planning script, `print_board`) | `X` internally, prints as **B** | `O` internally, prints as **W** ("White") |

Functionally consistent (X always maps to Black/-1, O always maps to
Brown/1) but the *labels* differ file to file. If you add a new script, standardize
on Black/Brown (matches the physical carrom coins actually used and the Gemini
vision prompt) to avoid confusing "White" with an actual white coin.

## 4. How the dataset is actually built

There is no committed script that drives the physical teleop recording session.
Only the **board-state planner**, `src/lerobot/scripts/ticTacToe/board_generator.py`,
which decides what the *other 8 cells* should look like while recording a demo
for one target (empty) cell.

For a given target cell, it generates **10 board configurations**, one per
piece-count scenario:

| Scenario (X, O) | Total pieces on board (excl. target cell) |
|---|---|
| (0, 0) | Empty board |
| (1, 0) | 1 |
| (1, 1) | 2 |
| (2, 1) | 3 |
| (2, 2) | 4 |
| (3, 2) | 5 |
| (3, 3) | 6 |
| (4, 3) | 7 |
| (4, 4) | 8 |

For each scenario, piece positions among the remaining 8 cells are picked with
`random.sample` (randomized placement, fixed count), and validated by
`is_valid_game_state()`:
- turn-order legality (`X count == O count` or `O count + 1`, since X always
  moves first),
- no pre-existing winning line on the board.

**So, concretely:** recording a demo for target cell (1,1) (top-left) does *not*
mean the other 8 cells are always empty. Only the `(0,0)` scenario's demo has a
fully empty board. The other 9 demos have increasingly cluttered boards (up to 8
pieces) placed randomly in the remaining cells. This is what "randomly
configured board state each time" in the blog post refers to.

Each episode is **one pick-and-place**: one Brown/O coin, one target cell. The
"9 positions x 10 demos = 90 episodes" dataset is 90 single-placement episodes,
not multi-move games.

**Known gap in `board_generator.py`** (their own TODO, line ~100): if a scenario
can't find a valid config within 1000 attempts, it's silently skipped rather than
retried. This could under-fill certain scenarios for certain target cells, more
likely near diagonal-heavy positions where higher piece counts collide with the
no-winning-line constraint more often.

## 5. Why the ACT policy ignores the camera

Attention-interpretability analysis (blog + `output/attention_analysis_results`)
found the trained policy attends almost entirely to **task instruction** and
**joint state**, and barely to **vision**. This lines up with the dataset design:
the robot's token color (Brown/O) never varies across any of the 90 episodes.
Only cell position and background clutter vary, so color carries zero
information and the model has no incentive to look at the image to distinguish
anything. If a future dataset trains the robot to play either color, expect
vision attention to matter a lot more, since color would then be
task-relevant.

## 6. Known code issues to fix

- **Hardcoded Gemini API key** in `src/lerobot/play_TicTacToe.py` line 41
  (`client = genai.Client(api_key="AIza...")`). Move to an environment variable
  before this repo goes anywhere public. Flagged, not yet fixed (deferred by
  request).
- **Duplicate import** of `hw_to_dataset_features` in `play_TicTacToe.py` (once
  from `lerobot.record`, once from `lerobot.datasets.utils`). Second shadows
  first, harmless but sloppy.
- **Unguarded regex match** in `call_policy()`: `matches[-1]` on
  `re.findall(r'Place at position \d+', ...)` will raise `IndexError` if the
  instruction string ever doesn't contain that pattern.
- **`src/lerobot/scripts/ticTacToe/TicTacToeAlgorithm.py`** is a stale duplicate
  of the `analyzeboard`/`minimax`/`CompTurn`/`print_board` logic now living in
  `play_TicTacToe.py`, with leftover debug `print()` calls inside `minimax`.
  Candidate for deletion, since the logic has since moved into the main script.
- **No null-check** on `get_grid_image()`'s return. If the camera read fails
  (`ret == False`), `image` is `None` and the next call
  (`crop_image`/`get_LLM_output`) will crash instead of failing gracefully.

## 7. Code map

- `src/lerobot/play_TicTacToe.py`: main game loop. Camera capture, then perspective
  warp, then Gemini vision call (board state), then minimax (best move), then ACT
  policy (execution). Entry point: `python -m lerobot.play_TicTacToe`.
- `src/lerobot/scripts/ticTacToe/board_generator.py`: board-state planner used
  when designing/recording the training dataset (see section 4).
- `src/lerobot/scripts/ticTacToe/TicTacToeAlgorithm.py`: stale scratch copy of
  the minimax logic (see section 6).
- `src/lerobot/scripts/ticTacToe/image_transformation_testing.py`: standalone
  test/calibration script for the camera crop, perspective-warp, and Gemini vision
  pipeline (same logic as embedded in `play_TicTacToe.py`).
- `src/lerobot/record.py`, `src/lerobot/scripts/train.py`: stock upstream
  LeRobot CLI tools, used unmodified for data collection (`lerobot-record`) and
  training (`lerobot-train`). No project-specific wrapper script exists for
  either step.
- ACT policy itself (`src/lerobot/policies/act/`): modified upstream to accept
  a task-instruction embedding so one policy generalizes across all 9 grid-cell
  tasks (per README "Code Contributions"). Not re-verified line-by-line in this
  pass, worth a follow-up read if you're touching the policy.

## 8. Pipeline summary (per robot turn)

1. Capture image from fixed camera (`camera_index = 2` in current setup).
2. Crop to board region, perspective-warp to top-down view, crop/rotate to final
   framing (hardcoded percentages/points in `play()`, calibrated to one specific
   physical camera rig; will need recalibrating if the camera or table setup
   changes).
3. Send image to Gemini (`gemini-2.0-flash`) with a fixed prompt asking for
   per-cell state (`Empty`/`Brown`/`Black`). This is perception, not reasoning.
4. Parse the text response into a 9-element vector (`-1`/`0`/`1`).
5. Run `minimax`/`CompTurn` locally (plain game-tree search, no ML) to pick the
   robot's best cell.
6. Build instruction string `"Place at Position N"`, hand off to the trained ACT
   policy via `call_policy()`, which drives the robot for up to
   `robot_turn_time_s` seconds executing the pick-and-place.
7. Text-to-speech (`log_say`) announces turns and game-over via
   `cfg.play_sounds`.
