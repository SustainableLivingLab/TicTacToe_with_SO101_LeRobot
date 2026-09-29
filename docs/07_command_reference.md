# 07. Command Reference

End to end: installing LeRobot, setting up hardware, recording a dataset,
training, and running inference. All commands are `python -m ...` module
invocations; this repo has no installed CLI aliases (no `lerobot-record` /
`lerobot-train` console scripts in `pyproject.toml`).

## Key concepts

A handful of values show up repeatedly across every command below. Skim
this before running anything; each one is explained again in context where
it first appears, but this is the plain-language version up front.

- **`--robot.port=` / `--teleop.port=`** (a COM port on Windows, e.g.
  `COM5`, or a device path on Linux/Mac, e.g. `/dev/ttyACM0`): which USB
  connection the follower arm (`robot`) or leader arm (`teleop`) is plugged
  into. Found with `python -m lerobot.find_port` (section 2 below). This
  can change if you unplug and replug the arm into a different USB port.

- **`--robot.id=` / `--teleop.id=`** (e.g. `my_follower_arm`): a nickname
  you make up yourself for each arm, not something read off the hardware
  or printed on it anywhere. You pick this name the first time you
  calibrate that arm (section 2), and it names the calibration file saved
  for it. From then on, you reuse that exact same name in every later
  command (recording, inference) whenever you refer to that same physical
  arm. If you forget what you named it, check the calibration files saved
  during setup, or just recalibrate and pick a name again.

- **Camera index** (a plain number: `0`, `1`, `2`, ...): which physical
  camera a given number refers to, assigned by your computer, not
  something printed on the camera or visible in Device Manager. With two
  cameras connected, one might be `0` and the other `1`, or `2` and `3`,
  there's no way to know without testing. Section 3 below has a small
  script (`find_cameras.py`) that opens each index and shows a live
  preview, so you can see which physical camera each number is.

- **Camera name** (e.g. `front`, `top`): unlike the index above, this part
  is not assigned by anything, it's a label you choose. Call your two
  cameras whatever you want, `front`/`top`, `cam1`/`cam2`, anything, the
  only rule is you must use the exact same names both when recording data
  and later when running inference, since the trained policy expects
  observations under those specific names.

- **`--dataset.repo_id=`** (e.g. `melissa/tictactoe-position-3`): the name
  of a dataset on the Hugging Face Hub, in the form
  `<your-huggingface-username>/<dataset-name>`. You do not need to create
  this on huggingface.co first, recording creates it automatically. See
  section 3, "What to name each dataset's `repo_id`," for the full
  breakdown.

- **`--policy.path=`**: where to find a trained model checkpoint, either a
  local folder path on the machine you're running this command on, or a
  Hugging Face Hub repo id (like `dataset.repo_id` above, but for a model
  instead of a dataset) that gets downloaded automatically. This only
  matters once you have a trained policy, at inference time (section 6).

- **`--config_path=<file>.yaml`**: an alternative to typing every
  `--flag=value` individually on the command line. Point this at a YAML
  file (like `train_config.yaml`, included in this repo) that holds the
  same settings, useful when a value (like a list of 9 dataset repo ids)
  is awkward to type as a single command-line flag. See section 4.

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

### Hugging Face login (one-time)

Recording pushes each dataset to the Hugging Face Hub automatically when
the session ends (`DatasetRecordConfig.push_to_hub` defaults to `True` in
`TicTacToe_with_SO101/src/lerobot/record.py`). You need to be logged in
first, once per machine:

```bash
huggingface-cli login
```

Paste a Hub access token when prompted (create one at
https://huggingface.co/settings/tokens if you don't have one, "Write"
permission). You do **not** need to manually create the dataset repo on
huggingface.co first; `dataset.push_to_hub()` calls `create_repo()`
internally and creates it automatically the first time each `repo_id` is
pushed.

### What to name each dataset's `repo_id`

`--dataset.repo_id=<your-username>/tictactoe-position-<N>` has two parts:

- `<your-username>` is your actual Hugging Face username (the account you
  just logged into above), not a literal placeholder and not this
  project's username. Find it at https://huggingface.co/settings/profile
  or from the URL of your own Hub profile page.
- `tictactoe-position-<N>` is a naming convention, not a required exact
  string. `<N>` is the grid cell this recording session is for (1 through
  9). The name after your username can be anything; `tictactoe-position-N`
  is just a clear, consistent pattern, one repo per cell, that this doc
  and `train_config.yaml`'s placeholders use throughout. If you use a
  different naming pattern, use it consistently and update
  `train_config.yaml`'s 9 entries to match your real names.

Concretely: if your Hugging Face username is `melissa`, recording for cell
3 uses `--dataset.repo_id=melissa/tictactoe-position-3`, and that becomes a
new dataset repo at `https://huggingface.co/datasets/melissa/tictactoe-position-3`
after the session finishes and pushes.

### Recording command

This project uses **two cameras**, front-angled and top-down (see
`01_hardware_setup.md`, "Camera rig"). Both must be listed in
`--robot.cameras=` so both get recorded into the dataset and both are what
the ACT policy trains on. `front` and `top` are just labels this doc uses
consistently; you can name them anything, as long as the same names are
used again at inference (step 6).

### Finding each camera's index

The index each camera needs (`index_or_path` in the command below) is a
plain integer (`0`, `1`, `2`, ...) that OpenCV assigns per connected
camera, in an order that is not predictable from Device Manager or a cable
label. Do not guess it; find it by actually opening each index and looking
at the image. Save this as `find_cameras.py` and run it:

```python
import cv2

for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Index {i}: working, frame shape {frame.shape}")
            cv2.imshow(f"Camera index {i}", frame)
            cv2.waitKey(1500)
            cv2.destroyAllWindows()
        cap.release()
    else:
        print(f"Index {i}: not available")
```

```bash
python find_cameras.py
```

Each working index pops up a preview window for about 1.5 seconds so you
can see which physical camera it is. Note which index shows the
front-angled view and which shows the top-down view, then use those two
numbers as `index_or_path` for `front` and `top` respectively in the
command below. Indices can shift if you unplug/replug a camera or reboot;
re-run this script if a camera stops responding at its previous index.

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
| `--dataset.repo_id` | Dataset identifier. `<your-username>/<dataset-name>` on the Hugging Face Hub; see "What to name each dataset's `repo_id`" above. |
| `--dataset.single_task` | Task string stored with every frame in this recording session. For this project: `"Place at Position N"`, matching what `play_TicTacToe.py` sends at inference. |
| `--dataset.num_episodes` | Number of episodes to record in this run. This project recorded 10 per grid cell. |
| `--dataset.root` | Local storage path, if not using the default cache location. Recording still pushes to the Hub unless `--dataset.push_to_hub=false` is also set. |

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

`TicTacToe_with_SO101/train_config.yaml` in this repo is a ready-made
template for exactly this:

```yaml
dataset:
  repo_id:
    - your-username/tictactoe-position-1
    - your-username/tictactoe-position-2
    - your-username/tictactoe-position-3
    - your-username/tictactoe-position-4
    - your-username/tictactoe-position-5
    - your-username/tictactoe-position-6
    - your-username/tictactoe-position-7
    - your-username/tictactoe-position-8
    - your-username/tictactoe-position-9
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

Edit the 9 `your-username/tictactoe-position-N` placeholders to your real
repo ids from step 3, then run:

```bash
python -m lerobot.scripts.train --config_path=train_config.yaml
```

This is the definitive way to pass all 9 datasets in one run.

Key `TrainPipelineConfig` fields (`TicTacToe_with_SO101/src/lerobot/configs/train.py`):

| Field | Meaning |
|---|---|
| `dataset.repo_id` | Dataset(s) to train on. List of the 9 repo ids from step 3, in the YAML config as shown above (see step 3, "What to name each dataset's `repo_id`", for what these strings actually are). |
| `policy.type` | Policy class. Use `act_lang` for this project (see `06_act_configuration.md`). |
| `--policy.<field>` | Any field on `ACTLangConfig` (layer counts, `dim_model`, optimizer learning rate, etc.), see `06_act_configuration.md` for the full list. |
| `--batch_size` | This project used 64 (larger than ACT's typical default), which reduced the number of steps needed for convergence. |
| `--steps` | This project used 25,000 (40 epochs). A checkpoint at 15,000 performed comparably. |
| `--output_dir` | Where checkpoints are written. |
| `--wandb.enable` | Enable Weights & Biases logging (requires `wandb login` once). |
| `--resume` | Resume from `--output_dir`'s last checkpoint. |

Checkpoints land in `<output_dir>/checkpoints/<step>/pretrained_model/`,
containing `config.json`, `model.safetensors`, `train_config.json`.

### Training without a local GPU (Modal)

`python -m lerobot.scripts.train` is local compute; it does not run in the
cloud by itself. If you have no local GPU, run the exact same command on a
rented GPU via [Modal](https://modal.com/docs/guide) instead of your own
machine.

Install and authenticate Modal locally first (this only submits the job,
Modal runs the training itself):

```bash
pip install modal
modal setup
```

Create `modal_train.py` in the repo root:

```python
import modal

app = modal.App("tictactoe-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .run_commands(
        "git clone https://github.com/SustainableLivingLab/TicTacToe_with_SO101_LeRobot.git /repo"
    )
    .workdir("/repo/TicTacToe_with_SO101")
    .run_commands("pip install -e .")
)

checkpoints = modal.Volume.from_name("tictactoe-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/checkpoints": checkpoints},
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train():
    import subprocess

    subprocess.run(
        [
            "python", "-m", "lerobot.scripts.train",
            "--config_path=train_config.yaml",
            "--output_dir=/checkpoints/tictactoe_act_lang",
        ],
        check=True,
    )
    checkpoints.commit()
```

`train_config.yaml` is the same file described above
(`TicTacToe_with_SO101/train_config.yaml`, already in the repo). The
version Modal clones must have your real repo ids filled in, not the
placeholders, so edit it and push the change before running `modal run`,
or add a `run_commands` step in the image that overwrites it with your
edited version.

### GPU choice and timing

`gpu="A10G"` in the script above is one option among several Modal offers;
change this string to switch GPU tier. At the default 25,000 steps,
batch size 64 (this project's own settings, see step 4's config table),
one real run on A10G took about 4 hours. A100 and H100 are faster per step
but do not reduce total time enough to fit a short session; going from
A10G to A100 to H100 is roughly a 1.5-2x speedup each step, not 10x, so a
~4 hour A10G run is still well over an hour even on H100. If you are
time-boxed (for example, a 2-3 hour workshop where students record data
and train live), the effective lever is `steps`, not GPU tier; see "Faster
training for a time-boxed session" below. Real per-second Modal GPU rates
as of this writing: A10 (`gpu="A10G"`) about $1.10/hr, A100 80GB
(`gpu="A100"`) about $2.50/hr, H100 (`gpu="H100"`) about $3.95/hr,
significantly more if you pin a specific region. New Modal accounts get
free starting credit, check https://modal.com/pricing for current terms.

### Faster training for a time-boxed session

This project's own training notes (`03_dataset_and_training.md`) record
that a checkpoint at 15,000 steps performed comparably to the full 25,000
step run. Lowering `--steps` (or `steps:` in `train_config.yaml`) to
15,000 cuts training time roughly proportionally, since ACT training time
scales close to linearly with step count at a fixed batch size. Going
meaningfully lower than 15,000 has not been tested by this project; treat
it as an unverified tradeoff, more likely to produce a visibly weaker
policy the further you drop, not a guaranteed-safe shortcut. If a session
is hard-capped at a specific duration, the reliable approach is to
estimate from a known data point (this project's own ~4 hours for 25,000
steps on A10G) rather than assume any GPU tier alone reaches a target time.

`modal.Secret.from_name("huggingface-secret")` needs a Modal secret holding
your `HF_TOKEN`, created once via the Modal dashboard or
`modal secret create huggingface-secret HF_TOKEN=<your token>`, since the
dataset pull from Hugging Face Hub needs authentication the same way it
would locally.

Run it:

```bash
modal run modal_train.py
```

`--output_dir=/checkpoints/tictactoe_act_lang` writes checkpoints into the
Modal Volume (`checkpoints.commit()` persists them after the run), not into
the container's throwaway local disk. Download the trained checkpoint back
to whatever machine will run inference:

```bash
modal volume get tictactoe-checkpoints tictactoe_act_lang/checkpoints/last/pretrained_model ./pretrained_model
```

Then point `--policy.path=./pretrained_model` at that downloaded folder in
the inference command (step 6), or `huggingface-cli upload` it to the Hub
and reference the Hub repo id instead, same as any other cloud-trained
checkpoint (see step 6's note on this).

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
machine running this command (the one connected to the robot). It is not
automatically the same machine training ran on. Two concrete options:

- **Local path**, if you copied or downloaded the checkpoint folder onto
  this machine, e.g.
  `--policy.path=outputs/train/tictactoe_act_lang/checkpoints/last/pretrained_model`
  (matches `--output_dir=` from step 4) or, for the Modal path from step 4,
  wherever you ran `modal volume get` to, e.g.
  `--policy.path=./pretrained_model`.
- **Hugging Face Hub repo id**, if you uploaded the checkpoint with
  `huggingface-cli upload` (see step 4, "Upload policy checkpoints" link),
  e.g. `--policy.path=melissa/tictactoe-act-lang`. This downloads the
  checkpoint automatically; no manual file transfer needed.

Training (step 4) is local compute by default; it does not run in the
cloud unless you explicitly run it on a rented GPU or Modal (see step 4,
"Training without a local GPU (Modal)"). Either way, the checkpoint has to
reach the robot machine one of the two ways above before this command can
use it.

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
