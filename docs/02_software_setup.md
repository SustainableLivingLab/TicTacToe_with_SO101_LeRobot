# 02. Software Setup

## Base install

Follow LeRobot's own installation guide first:
`TicTacToe_with_SO101/docs/source/installation.mdx` (or the upstream docs at
https://huggingface.co/docs/lerobot). Summary:

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge
pip install -e .
```

For SO-101 motor communication, also install the Feetech extra:

```bash
pip install -e ".[feetech]"
```

## Project-specific dependencies

This project additionally uses:

- `opencv-python` (`cv2`): camera capture and perspective transforms.
- `Pillow`: image handling.
- `google-genai`: Gemini API client for board-state vision.

## Environment variables

The game script requires a Gemini API key, read from the environment:

```bash
GEMINI_API_KEY=<your key>
```

Set it in your shell, or copy `TicTacToe_with_SO101/.env.example` to
`TicTacToe_with_SO101/.env` and fill in the real key. `.env` is gitignored;
never commit a real key. The script raises an error at startup if
`GEMINI_API_KEY` is unset.

Note: the script reads `os.environ` directly and does not auto-load `.env`
files. Export the variable into your shell before running, or use your
IDE/run configuration's env-file support.

## Running the game

```bash
python -m lerobot.play_TicTacToe \
    --robot.type=so101_follower \
    --robot.port=<follower port> \
    --robot.id=<follower id> \
    --policy.path=<path to trained ACT checkpoint>
```

See `04_gameplay_pipeline.md` for what this actually runs.

## Recording a new dataset

Data collection uses LeRobot's standard `lerobot-record` CLI, unmodified.
There is no project-specific wrapper script. The board-state planner
(`TicTacToe_with_SO101/src/lerobot/scripts/ticTacToe/board_generator.py`)
decides what background pieces to place before each recorded episode; see
`03_dataset_and_training.md` for the scenarios it generates.

## Training the policy

Training uses LeRobot's standard `lerobot-train` CLI, unmodified, against a
recorded dataset. See `03_dataset_and_training.md` for the parameters used
for the current checkpoint.
