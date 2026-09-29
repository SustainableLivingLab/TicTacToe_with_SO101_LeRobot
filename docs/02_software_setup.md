# 02. Software Setup

## Clone this repo

This is not plain upstream LeRobot. `play_TicTacToe.py`, `board_generator.py`,
`image_transformation_testing.py`, the `act_lang` policy variant, and every
doc in this `docs/` folder exist only in this fork; they are not part of a
plain `git clone` of `huggingface/lerobot`. Clone this repo specifically
before doing anything else:

```bash
git clone https://github.com/SustainableLivingLab/TicTacToe_with_SO101_LeRobot.git
cd TicTacToe_with_SO101_LeRobot/TicTacToe_with_SO101
```

All commands below assume you are inside that `TicTacToe_with_SO101`
directory (the one containing this repo's `pyproject.toml`), not a separate
plain-LeRobot clone.

## Base install

LeRobot's own installation guide
(`TicTacToe_with_SO101/docs/source/installation.mdx`, or the upstream docs
at https://huggingface.co/docs/lerobot) shows `python=3.10`. Ignore that
number; this repo's `pyproject.toml` requires `>=3.12`. Use the version
shown below, not the upstream guide's:

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install ffmpeg -c conda-forge
pip install -e .
```

Use Python 3.12. If `pip install -e .` errors with a Python version
conflict, you are not inside this repo's `TicTacToe_with_SO101` directory;
see "Clone this repo" above. On Windows, if `pip install -e .` tries to
build `numpy` or `torch` from source (a `.tar.gz` download instead of a
`.whl`, followed by a `meson`/compiler error) instead of failing with a
version conflict, delete the conda env and recreate it with
`python=3.12` exactly as shown above, rather than installing a C compiler.

If `conda install` hangs at `Solving environment:` for more than a minute or
two, that is conda's classic solver stalling, not a network issue. Install
the faster solver once and retry:

```bash
conda install -n base conda-libmamba-solver -c conda-forge
conda install ffmpeg -c conda-forge --solver=libmamba
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

## Running the game, recording, and training

Full command reference (hardware bring-up, recording, training, inference)
is in `07_command_reference.md`. Data collection and training both use
LeRobot's standard `python -m lerobot.record` / `python -m lerobot.scripts.train`
modules, unmodified aside from the `act_lang` policy type (see
`06_act_configuration.md`). There is no project-specific wrapper script for
either step.
