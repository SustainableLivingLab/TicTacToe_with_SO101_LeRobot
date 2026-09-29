# 07. Command Reference

End to end: installing LeRobot, setting up hardware, recording a dataset,
training, and running inference. All commands are `python -m ...` module
invocations; this repo has no installed CLI aliases (no `lerobot-record` /
`lerobot-train` console scripts in `pyproject.toml`).

## 1. Install

Clone this repo first, not plain upstream LeRobot; the TicTacToe game loop,
`act_lang` policy, and dataset-planning scripts exist only here:

```bash
git clone https://github.com/SustainableLivingLab/TicTacToe_with_SO101_LeRobot.git
cd TicTacToe_with_SO101_LeRobot/TicTacToe_with_SO101
```

Run every command below from inside that `TicTacToe_with_SO101` directory.

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install ffmpeg -c conda-forge
pip install -e .
pip install -e ".[feetech]"   # STS3215 motor communication for SO-101
```

Use Python 3.12, exactly as shown above. If `conda install` hangs at
`Solving environment:` for more than a minute, see `02_software_setup.md`
for the libmamba solver fix. If `pip install -e .` tries to compile `numpy`
or `torch` from source instead of using a prebuilt wheel, delete the conda
env and recreate it with `python=3.12`; see `02_software_setup.md`.

Project-specific extras (Gemini vision client, if not already pulled in by
`pyproject.toml`):

```bash
pip install google-genai opencv-python
```

Set the Gemini API key (see `02_software_setup.md`):

```bash
export GEMINI_API_KEY=<your key>
```

## 2. Hardware bring-up

Full detail in `01_hardware_setup.md`. Command sequence:

```bash
# Find each arm's USB port
python -m lerobot.find_port

# Set motor IDs and baudrate (once per arm, after wiring motors individually)
python -m lerobot.setup_motors --robot.type=so101_follower --robot.port=<port>
python -m lerobot.setup_motors --teleop.type=so101_leader --teleop.port=<port>

# Calibrate (aligns leader/follower joint-position values)
python -m lerobot.calibrate --robot.type=so101_follower --robot.port=<port> --robot.id=<name>
python -m lerobot.calibrate --teleop.type=so101_leader --teleop.port=<port> --teleop.id=<name>
```

## 3. Record a dataset

This project uses **two cameras**, front-angled and top-down (see
`01_hardware_setup.md`, "Camera rig"). Both must be listed in
`--robot.cameras=` so both get recorded into the dataset and both are what
the ACT policy trains on. Find each camera's OpenCV index first (plug in one
at a time, or check `ls /dev/video*` on Linux / Device Manager on Windows,
and confirm with a quick test capture) before filling in the command below.

```bash
python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port=<follower port> \
    --robot.id=<follower id> \
    --robot.cameras="{
        front: {type: opencv, index_or_path: <front camera index>, width: 640, height: 480, fps: 30},
        top: {type: opencv, index_or_path: <top camera index>, width: 640, height: 480, fps: 30}
    }" \
    --teleop.type=so101_leader \
    --teleop.port=<leader port> \
    --teleop.id=<leader id> \
    --dataset.repo_id=<your-username>/tictactoe-position-<N> \
    --dataset.num_episodes=10 \
    --dataset.single_task="Place at Position <N>"
```

Key `DatasetRecordConfig` / `DatasetConfig` fields
(`TicTacToe_with_SO101/src/lerobot/record.py`,
`TicTacToe_with_SO101/src/lerobot/configs/default.py`):

| Flag | Meaning |
|---|---|
| `--robot.cameras` | Camera dict, one entry per camera. Key names (`front`, `top` above) become the dataset's camera feature names, and must match what `--robot.cameras=` uses later at inference (step 6). |
| `--dataset.repo_id` | Dataset identifier (local folder name / Hugging Face Hub repo). |
| `--dataset.single_task` | Task string stored with every frame in this recording session. For this project: `"Place at Position N"`, matching what `play_TicTacToe.py` sends at inference. |
| `--dataset.num_episodes` | Number of episodes to record in this run. This project recorded 10 per grid cell. |
| `--dataset.root` | Local storage path, if not using the default cache location. |

Run this command once per grid cell (9 times total, `N` = 1 through 9),
each time as a separate recording session with a different `single_task`
string. Before recording each of the 10 episodes within one session,
arrange the board's background pieces to match that demo's layout in
`08_dataset_90_boards.md`, then start the episode and teleoperate one
pick-and-place of a Red/O tile into the target cell (see
`03_dataset_and_training.md` for the full dataset design and piece
constraints).

## 4. Train

Step 3 records one dataset per grid cell (9 separate `repo_id`s).
`make_dataset()` (`TicTacToe_with_SO101/src/lerobot/datasets/factory.py`,
line ~85) checks `isinstance(cfg.dataset.repo_id, str)` and, when it is not
a plain string, treats it as multiple repo ids and concatenates them via
`MultiLeRobotDataset`, keeping only the data keys common across all of
them. This confirms multi-dataset training is supported by the code, but
this repo has no working example of the exact CLI syntax for passing a list
to `--dataset.repo_id` (draccus list syntax is typically
`--dataset.repo_id='[a, b, c]'`, unverified for this exact field). Verify
against `python -m lerobot.scripts.train --help` or a small test run before
relying on it, or construct the config in Python instead of via the CLI to
avoid the ambiguity.

```bash
python -m lerobot.scripts.train \
    --dataset.repo_id=<your-username>/tictactoe-position-1 \
    --policy.type=act_lang \
    --policy.device=cuda \
    --batch_size=64 \
    --steps=25000 \
    --save_freq=5000 \
    --output_dir=outputs/train/tictactoe_act_lang \
    --job_name=tictactoe_act_lang \
    --wandb.enable=true
```

Single-`repo_id` form shown above is confirmed correct. Replace with all 9
repo ids once the list syntax above is verified working in your
environment.

Key `TrainPipelineConfig` fields (`TicTacToe_with_SO101/src/lerobot/configs/train.py`):

| Flag | Meaning |
|---|---|
| `--dataset.repo_id` | Dataset to train on. A single repo id is confirmed to work; multiple repo ids are supported by `make_dataset()` but the exact CLI list syntax is not verified in this repo, see note above. |
| `--policy.type` | Policy class. Use `act_lang` for this project (see `06_act_configuration.md`). |
| `--policy.<field>` | Any field on `ACTLangConfig` (layer counts, `dim_model`, optimizer learning rate, etc.), see `06_act_configuration.md` for the full list. |
| `--batch_size` | This project used 64 (larger than ACT's typical default), which reduced the number of steps needed for convergence. |
| `--steps` | This project used 25,000 (40 epochs). A checkpoint at 15,000 performed comparably. |
| `--output_dir` | Where checkpoints are written. |
| `--wandb.enable` | Enable Weights & Biases logging (requires `wandb login` once). |
| `--resume` | Resume from `--output_dir`'s last checkpoint. |

Checkpoints land in `<output_dir>/checkpoints/<step>/pretrained_model/`,
containing `config.json`, `model.safetensors`, `train_config.json`.

## 5. Evaluate a checkpoint (no physical robot)

```bash
python -m lerobot.scripts.eval \
    --policy.path=<output_dir>/checkpoints/last/pretrained_model \
    --env.type=<simulation env, if applicable> \
    --eval.n_episodes=10
```

Not typically used for this project since the task is real-hardware only,
no simulation environment was built for Tic-Tac-Toe. Included here for
completeness; see LeRobot's own docs for simulation-based evaluation.

## 6. Run inference (play a game against the robot)

```bash
python -m lerobot.play_TicTacToe \
    --robot.type=so101_follower \
    --robot.port=<follower port> \
    --robot.id=<follower id> \
    --robot.cameras="{
        front: {type: opencv, index_or_path: <front camera index>, width: 640, height: 480, fps: 30},
        top: {type: opencv, index_or_path: <top camera index>, width: 640, height: 480, fps: 30}
    }" \
    --policy.path=<output_dir>/checkpoints/last/pretrained_model \
    --robot_turn_time_s=30 \
    --player_turn_time_s=10 \
    --fps=30
```

`--robot.cameras=` here must use the same camera key names (`front`, `top`)
used when recording the dataset in step 3, since the ACT policy expects an
observation with those exact feature names. This is separate from the
Gemini board-reading step, which captures independently via its own
hardcoded `camera_index = 2` inside `play_TicTacToe.py` (see
`01_hardware_setup.md`, "Camera rig", and `04_gameplay_pipeline.md`); it is
not controlled by this flag.

Before running: place the physical board with the human's first X already
placed (see `03_dataset_and_training.md`, "Turn assignment", the script does
not wait for a first human move on its own). `GEMINI_API_KEY` must be set in
the environment (see `02_software_setup.md`).

`TicTacToeConfig` fields
(`TicTacToe_with_SO101/src/lerobot/play_TicTacToe.py`):

| Flag | Meaning |
|---|---|
| `--robot.*` | Robot connection config, same shape as `record`/`calibrate`. |
| `--policy.path` | Path (local or Hub) to the trained ACT checkpoint. |
| `--play_sounds` | Enable/disable text-to-speech turn announcements. Default `True`. |
| `--fps` | Control loop rate during policy execution. Default 30. |
| `--robot_turn_time_s` | Max seconds the policy runs per robot turn. Default 30. |
| `--player_turn_time_s` | Seconds the script waits for the human's physical move. Default 10. |
| `--use_videos` | Whether the mock dataset metadata built for `make_policy` encodes frames as video. Default `True`. Does not affect the trained checkpoint's own dataset format. |

See `04_gameplay_pipeline.md` for what the script does turn by turn.

## Full run order, summarized

```
install -> hardware bring-up -> calibrate -> record (x9 cells) -> train -> play
```
