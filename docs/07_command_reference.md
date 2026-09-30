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

- **`--dataset.repo_id=`** (e.g. `your-username/tictactoe-position-3`): the name
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
  same settings, easier to review and reuse than one very long command
  line for training. See section 4.

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

### One dataset, not nine

Earlier versions of this doc said to create 9 separate dataset repos (one
per grid cell) and combine them at training time via a list of repo ids.
That does not work in this lerobot version:
`TrainPipelineConfig.validate()` (`TicTacToe_with_SO101/src/lerobot/configs/train.py`,
line ~107) raises `NotImplementedError("LeRobotMultiDataset is not
currently implemented.")` the moment `dataset.repo_id` is a list, before
training even starts, and `make_dataset()`
(`TicTacToe_with_SO101/src/lerobot/datasets/factory.py`, line ~99) has the
same hard `raise` guarding its multi-dataset branch. Multi-dataset training
is not available in this codebase, full stop; it is present as dead code,
not a working feature.

The real workflow: record all 9 cells into **one single dataset repo**,
using `--resume=true` for the 2nd through 9th recording sessions so each
new session appends its episodes to the same growing dataset instead of
creating a separate one. `single_task` is attached per-frame at record
time (`record.py`, `dataset.add_frame(frame, task=single_task)`), so each
session can use its own `"Place at Position N"` string even though they
all land in the same dataset.

**If you already recorded 9 separate datasets** (one per cell, following
the earlier, incorrect version of this doc), you do not need to re-record.
`TicTacToe_with_SO101/merge_datasets.py`, included in this repo, downloads
all 9 existing dataset repos and merges them into one new dataset repo you
can point `train_config.yaml` at. Verified working against real recorded
data (9 real datasets, 90 episodes). Read the script's own docstring
before running it, in particular the note on running it with a *newer*
lerobot install (its own venv, `pip install "lerobot[dataset]"`), not this
repo's own forked lerobot, since this repo's fork cannot read dataset
format v3.0 (see the docstring for the exact `ForwardCompatibilityError`
this causes if run in the wrong environment). It also includes a required
small test-run step first, always do that before the full merge.

If your source datasets' task strings do not already match the
`"Place at Position N"` format `play_TicTacToe.py` sends at inference (for
example, if they were recorded as `"tictactoe-position-N"` by a tool like
LeLab), pass `--rewrite-task auto-position` to have the script rewrite
each episode's task string to the correct format automatically as part of
the merge.

`--dataset.repo_id=<your-username>/tictactoe` has two parts:

- `<your-username>` is your actual Hugging Face username (the account you
  just logged into above), not a literal placeholder and not this
  project's username. Find it at https://huggingface.co/settings/profile
  or from the URL of your own Hub profile page.
- `tictactoe` (or any name you like) is the dataset name. Pick one name and
  reuse the exact same `repo_id` across all 9 recording sessions; do not
  vary it per cell.

Concretely: if your Hugging Face username is `alex`, every one of the 9
recording sessions uses `--dataset.repo_id=alex/tictactoe`, and that
becomes one dataset repo at
`https://huggingface.co/datasets/alex/tictactoe` holding all 90
episodes once the 9th session finishes and pushes.

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

**First session (cell 1)**, creates the dataset:

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
    --dataset.repo_id=<your-username>/tictactoe \
    --dataset.num_episodes=10 \
    --dataset.single_task="Place at Position 1"
```

**Sessions 2 through 9 (cells 2-9)**, append to the same dataset, add
`--resume=true` and change only `--dataset.single_task=`:

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
    --dataset.repo_id=<your-username>/tictactoe \
    --dataset.num_episodes=10 \
    --dataset.single_task="Place at Position 2" \
    --resume=true
```

`--dataset.repo_id` is identical across all 9 commands. Only
`--dataset.single_task` changes (`"Place at Position 1"` through
`"Place at Position 9"`), and `--resume=true` is added from the 2nd session
onward.

Key `DatasetRecordConfig` / `DatasetConfig` fields
(`TicTacToe_with_SO101/src/lerobot/record.py`,
`TicTacToe_with_SO101/src/lerobot/configs/default.py`):

| Flag | Meaning |
|---|---|
| `--robot.cameras` | Camera dict, one entry per camera. Key names (`front`, `top` above) become the dataset's camera feature names, and must match what `--robot.cameras=` uses later at inference (step 6). |
| `--dataset.repo_id` | Dataset identifier. `<your-username>/<dataset-name>` on the Hugging Face Hub, the same value for all 9 sessions; see "One dataset, not nine" above. |
| `--dataset.single_task` | Task string attached to every frame recorded in this session. Changes each of the 9 sessions: `"Place at Position N"`, matching what `play_TicTacToe.py` sends at inference. |
| `--dataset.num_episodes` | Number of episodes to record in this session. This project recorded 10 per grid cell. |
| `--resume` | `true` for the 2nd through 9th sessions, so this session's episodes append to the existing dataset at `repo_id` instead of trying to create a new one (which would fail, the repo already exists after session 1). Omit or `false` only for the very first session. |
| `--dataset.root` | Local storage path, if not using the default cache location. Recording still pushes to the Hub unless `--dataset.push_to_hub=false` is also set. |

Run one of the two commands above once per grid cell (9 times total).
Before recording each of the 10 episodes within one session, arrange the
board's background pieces to match that demo's layout in
`08_dataset_90_boards.md`, then start the episode and teleoperate one
pick-and-place of a Red/O tile into the target cell (see
`03_dataset_and_training.md` for the full dataset design and piece
constraints).

## 4. Train

You train **once**. Step 3 now records all 9 cells into one single
dataset (`repo_id`), so training points at that one dataset, no
multi-dataset step needed. One training run over that dataset produces one
task-conditioned policy covering all 9 cells (see `06_act_configuration.md`,
the task-instruction mechanism is what makes one policy handle all 9
positions from a single dataset containing all 9 tasks).

`python -m lerobot.scripts.train` supports `--config_path=<file>` (see
`train.py`'s `@parser.wrap()`, `config_path` argument), which loads a YAML
config; any field can still be overridden with a normal `--flag=value` on
the same command. `TicTacToe_with_SO101/train_config.yaml` in this repo is
a ready-made template:

```yaml
dataset:
  repo_id: your-username/tictactoe
policy:
  type: act_lang
  device: cuda
  push_to_hub: false
batch_size: 64
steps: 25000
save_freq: 5000
output_dir: outputs/train/tictactoe_act_lang
job_name: tictactoe_act_lang
wandb:
  enable: true
```

Edit `dataset.repo_id` to the single dataset repo id from step 3, then run:

```bash
python -m lerobot.scripts.train --config_path=train_config.yaml
```

### What `ValueError: 'policy.repo_id' argument missing` means

If you see this error, here is what it actually means and why: after
training finishes, this script tries to automatically upload the trained
model to the Hugging Face Hub, the same way `dataset.push_to_hub` uploads
a recorded dataset. To upload something, it needs a name to upload it
under, a `policy.repo_id` (e.g. `your-username/tictactoe-act-lang`). This
model-upload behavior is turned on by default
(`PreTrainedConfig.push_to_hub` defaults to `True` in
`TicTacToe_with_SO101/src/lerobot/configs/policies.py`), and if it's on
but no name was given, training refuses to start at all rather than train
for hours and then fail to upload at the very end.

`train_config.yaml`'s `policy: push_to_hub: false` simply turns this
upload off. The checkpoint still gets saved normally to `output_dir` on
disk either way; `push_to_hub` only controls whether it also gets copied
to the Hub automatically. If the automatic Hub upload is wanted, set
`push_to_hub: true` and add a `repo_id: your-username/<any-name>` line
under `policy:` in the YAML, and the error goes away because a name now
exists to upload under.

Key `TrainPipelineConfig` fields (`TicTacToe_with_SO101/src/lerobot/configs/train.py`):

| Field | Meaning |
|---|---|
| `dataset.repo_id` | The single dataset from step 3, containing all 9 cells' episodes. |
| `policy.type` | Policy class. Use `act_lang` for this project (see `06_act_configuration.md`). |
| `policy.push_to_hub` | Whether to push the trained checkpoint to the Hub automatically. `false` needs no `policy.repo_id`; `true` requires one (see above). |
| `policy.<field>` | Any field on `ACTLangConfig` (layer counts, `dim_model`, optimizer learning rate, etc.), see `06_act_configuration.md` for the full list. |
| `batch_size` | This project used 64 (larger than ACT's typical default), which reduced the number of steps needed for convergence. |
| `steps` | This project used 25,000 (40 epochs). A checkpoint at 15,000 performed comparably. |
| `output_dir` | Where checkpoints are written. |
| `wandb.enable` | Enable Weights & Biases logging (requires `wandb login` once). |
| `resume` | Resume from `output_dir`'s last checkpoint, if a training run was interrupted. |

Checkpoints land in `<output_dir>/checkpoints/<step>/pretrained_model/`,
containing `config.json`, `model.safetensors`, `train_config.json`.

### Training without a local GPU (Hugging Face Jobs)

`python -m lerobot.scripts.train` is local compute; it does not run in the
cloud by itself. If you have no local GPU, run the exact same command on a
rented GPU via [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)
instead of your own machine. Since the dataset already lives on the Hub,
this keeps everything (data, compute, and the trained model) inside
Hugging Face, no other cloud account needed.

Upstream lerobot has its own simpler flag for this, `lerobot-train
--job.target=<flavor>`, which submits a job without any manual `git
clone` step. **This project cannot use that flag.** It runs against HF's
own lerobot runtime image, which only has stock policy types; this
project's policy, `--policy.type=act_lang`, is custom code that exists
only in this fork's `TicTacToe_with_SO101/src/lerobot/policies/act/`, not
in any pip-installed lerobot. The approach below installs this fork's
actual code inside the job first, specifically so `act_lang` is available
to train with.

Install the `hf` CLI locally first (this only submits the job, Hugging
Face runs the training itself), and make sure you're logged in:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

Jobs need a positive credit balance on the account or organization
running them; see https://huggingface.co/docs/hub/jobs-pricing and
https://huggingface.co/settings/billing.

Run the job directly from the CLI, no separate script file needed. This
clones the repo, installs it, and runs training in one container:

```bash
hf jobs run \
    --flavor a10g-large \
    --timeout 6h \
    --secrets HF_TOKEN \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    bash -c "
        git clone https://github.com/SustainableLivingLab/TicTacToe_with_SO101_LeRobot.git /repo &&
        cd /repo/TicTacToe_with_SO101 &&
        pip install -e . &&
        python -m lerobot.scripts.train --config_path=train_config.yaml
    "
```

`train_config.yaml` is the same file described above
(`TicTacToe_with_SO101/train_config.yaml`, already in the repo). The job
clones the repo fresh from GitHub every time it runs, so it only ever sees
whatever is currently pushed to `main`, never a local, unpushed edit on
your own machine. **If you change `train_config.yaml` (for example, to set
`push_to_hub`, see below, or to lower `steps`), commit and push that
change before running `hf jobs run`,** or the job will train against the
old version of the file. To confirm what the job will actually use before
running it, check
https://github.com/SustainableLivingLab/TicTacToe_with_SO101_LeRobot/blob/main/TicTacToe_with_SO101/train_config.yaml
in a browser.

If you'd rather not edit and push the file at all, add a normal
`--flag=value` override at the end of the
`python -m lerobot.scripts.train` line inside the `bash -c "..."` block
instead, same as running it locally; a CLI override always takes priority
over whatever is in the file.

`--secrets HF_TOKEN` passes your logged-in Hugging Face token into the job
as an environment variable, needed to pull the dataset (and to push the
trained model, see below). It is encrypted server-side, not visible in
logs.

### GPU choice and timing

`--flavor a10g-large` in the command above is one option among many; see
the full table at
https://huggingface.co/docs/huggingface_hub/guides/jobs#select-the-hardware
(or run `hf jobs hardware` for the live list with current prices). At the
default 25,000 steps, batch size 64 (this project's own settings, see step
4's config table), a comparable A10G-class GPU run took about 4 hours in
practice (measured on a different cloud GPU provider, not HF Jobs
specifically, but the same GPU class). A100 and H100 flavors are faster
per step but do not reduce total time enough to fit a short session; going
from A10G to A100 to H100 is roughly a 1.5-2x speedup each step, not 10x,
so a ~4 hour A10G-class run is still well over an hour even on H200. If
you are time-boxed (for example, a 2-3 hour workshop where students record
data and train live), the effective lever is `steps`, not GPU tier; see
"Faster training for a time-boxed session" below. Real HF Jobs hourly
rates as of this writing: `a10g-large` $1.50/hr, `a100-large` $2.50/hr,
`h200` $5.00/hr; billing is per-second, so you only pay for what you use.
Jobs have a default 30-minute timeout, `--timeout 6h` above overrides that
for a long training run; adjust to your actual expected duration.

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
steps on an A10G-class GPU) rather than assume any GPU tier alone reaches
a target time.

### Getting the trained checkpoint onto the Hub

The cleanest way to have the trained model "sit in Hugging Face properly"
is to have the job itself push it to the Hub when training finishes,
instead of downloading it and re-uploading separately. Edit
`train_config.yaml` to add this, then commit and push the change (see the
reminder above: the job only sees what is actually pushed to `main`)
before running `hf jobs run`:

```yaml
policy:
  type: act_lang
  device: cuda
  push_to_hub: true
  repo_id: IndiaTechTeamSL2/tictactoe-act-lang
```

(See "What `ValueError: 'policy.repo_id' argument missing` means" above
for why both fields are needed together.) With this set, training uploads
the final checkpoint to `https://huggingface.co/IndiaTechTeamSL2/tictactoe-act-lang`
automatically, no manual step after the job completes. The `HF_TOKEN`
secret already passed into the job (see above) needs write access to that
`repo_id`'s namespace for the push to succeed.

If you'd rather review the checkpoint before publishing it, leave
`push_to_hub: false` (the repo's default), and retrieve it after the job
completes by mounting a Hub dataset repo as a writable volume for the
job's `output_dir`, or by having the job's command run
`huggingface-cli upload` as an explicit last step instead of relying on
`policy.push_to_hub`. Either way, once the checkpoint is a Hub repo id,
point `--policy.path=IndiaTechTeamSL2/tictactoe-act-lang` at it in the
inference command (step 6), same as any other cloud-trained checkpoint.

When `hf jobs run` starts, it prints a job URL and job ID in the terminal;
that URL is a browser page showing live logs and status, the simplest way
to watch training progress. To check on it later from a new terminal
instead, first list your jobs to find the id, then view that job's logs:

```bash
hf jobs ls
hf jobs logs <job_id>
```

`<job_id>` is the id shown in the `hf jobs ls` output (also visible at the
end of the URL printed when the job started), not a value you choose
yourself.

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
  (matches `--output_dir=` from step 4).
- **Hugging Face Hub repo id**, if the checkpoint was pushed to the Hub,
  either automatically by a Hugging Face Jobs training run (see step 4,
  "Getting the trained checkpoint onto the Hub") or manually with
  `huggingface-cli upload`, e.g.
  `--policy.path=IndiaTechTeamSL2/tictactoe-act-lang`. This downloads the
  checkpoint automatically; no manual file transfer needed.

Training (step 4) is local compute by default; it does not run in the
cloud unless you explicitly run it on a rented GPU or via Hugging Face
Jobs (see step 4, "Training without a local GPU (Hugging Face Jobs)").
Either way, the checkpoint has to reach the robot machine one of the two
ways above before this command can use it.

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
