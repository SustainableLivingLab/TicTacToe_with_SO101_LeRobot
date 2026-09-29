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

`<name>` (the `--robot.id=` / `--teleop.id=` value) is not read from the
hardware; it is a name you invent yourself, e.g. `my_follower_arm`. It is
used only to name the calibration file written to disk for that arm
(`RobotConfig.id` in `TicTacToe_with_SO101/src/lerobot/robots/config.py`).
Whatever you type here during calibration is the `--robot.id=` /
`--teleop.id=` value to reuse in every later command (recording, training
input, and inference) for that same physical arm.

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

You train **once**, not once per cell. Step 3 records one dataset per grid
cell (9 separate `repo_id`s), but all 9 feed a single training run that
produces one task-conditioned policy covering all 9 cells (see
`06_act_configuration.md`, the task-instruction mechanism is what makes one
policy handle all 9 positions). Training once per cell would defeat that
and give you 9 separate, non-cell-aware policies instead.

`make_dataset()` (`TicTacToe_with_SO101/src/lerobot/datasets/factory.py`,
line ~85) accepts a list of repo ids for `dataset.repo_id` and concatenates
them automatically via `MultiLeRobotDataset`. To pass a list reliably (CLI
flag quoting for a list is inconsistent across shells), use a YAML config
file instead of a CLI flag for this one field. `python -m lerobot.scripts.train`
supports `--config_path=<file>` (see `train.py`'s `@parser.wrap()`,
`config_path` argument), which loads a YAML config, and any field can still
be overridden or added on top of it as a normal `--flag=value` on the same
command.

Create `train_config.yaml`:

```yaml
dataset:
  repo_id:
    - <your-username>/tictactoe-position-1
    - <your-username>/tictactoe-position-2
    - <your-username>/tictactoe-position-3
    - <your-username>/tictactoe-position-4
    - <your-username>/tictactoe-position-5
    - <your-username>/tictactoe-position-6
    - <your-username>/tictactoe-position-7
    - <your-username>/tictactoe-position-8
    - <your-username>/tictactoe-position-9
policy:
  type: act_lang
  device: cuda
batch_size: 64
steps: 25000
save_freq: 5000
output_dir: outputs/train/tictactoe_act_lang
job_name: tictactoe_act_lang
wandb:
  enable: true
```

Then run:

```bash
python -m lerobot.scripts.train --config_path=train_config.yaml
```

This is the definitive way to pass all 9 datasets in one run. Replace the 9
placeholder repo ids with your own from step 3.

Key `TrainPipelineConfig` fields (`TicTacToe_with_SO101/src/lerobot/configs/train.py`):

| Field | Meaning |
|---|---|
| `dataset.repo_id` | Dataset(s) to train on. List of 9 repo ids in the YAML config, as shown above, for this project's full multi-cell training run. |
| `policy.type` | Policy class. Use `act_lang` for this project (see `06_act_configuration.md`). |
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
observation with those exact feature names. These key names are arbitrary,
`front` and `top` are just the names used throughout this doc; any names
work as long as they match between recording and inference. This is
separate from the Gemini board-reading step, which captures independently
via its own hardcoded `camera_index = 2` inside `play_TicTacToe.py` (see
`01_hardware_setup.md`, "Camera rig", and `04_gameplay_pipeline.md`); it is
not controlled by this flag.

`--policy.path=` must point to a checkpoint that actually exists on the
machine running this command. Training (step 4) is local compute; it does
not run "in the cloud" by itself. If you have no local GPU, train on a
rented cloud GPU or Google Colab instead (see
`TicTacToe_with_SO101/docs/source/il_robots.mdx`, "Train using Colab" and
"Upload policy checkpoints", for a worked example including the exact
`huggingface-cli upload` command). Either way, the checkpoint then needs to
reach the machine connected to the robot: copy
`<output_dir>/checkpoints/last/pretrained_model` there directly and point
`--policy.path=` at that local copy, or upload it to the Hugging Face Hub
from the training machine and set `--policy.path=<hf_user>/<repo_name>` on
the robot machine instead, which downloads it automatically.

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
