"""
Drives export_dataset_to_v2_1.py one episode at a time, alternating
between two venvs (a newer-lerobot venv for reading, this fork's own venv
for writing), so the whole dataset is never staged on disk at once. See
export_dataset_to_v2_1.py's own module docstring for why this exists.

This script itself does not import lerobot at all; run it with any Python
that has nothing more than the standard library (it just shells out to the
two venvs' own python.exe via subprocess).

Usage (from TicTacToe_with_SO101/, after both venvs exist):

    python export_orchestrate.py \
        --source IndiaTechTeamSL2/tictactoe \
        --dest IndiaTechTeamSL2/tictactoe-v2_1 \
        --read-python .export_venv_read/Scripts/python.exe \
        --write-python .export_venv_write/Scripts/python.exe \
        --rewrite-task auto-position

Test on a small slice first:

    python export_orchestrate.py \
        --source IndiaTechTeamSL2/tictactoe \
        --dest IndiaTechTeamSL2/tictactoe-v2_1-test \
        --read-python .export_venv_read/Scripts/python.exe \
        --write-python .export_venv_write/Scripts/python.exe \
        --rewrite-task auto-position \
        --limit-episodes 2

Resumable: if interrupted partway through, re-running the same command
picks up from the first episode whose entry is not yet marked done in
--state-path (default ./export_state.json), without re-reading or
re-writing episodes already completed. The local (not yet pushed) v2.1
dataset under --local-root is left in place between runs for this reason.

Disk usage at any point in time: roughly one episode's raw pickle (a few
GB, deleted immediately after that episode is written) plus the
video-encoded local dataset under --local-root (grows as episodes are
added, but video-encoded frames are much smaller than raw pickled ones).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def get_num_episodes(read_python: str, source: str, episode_index_probe: int = 0) -> int:
    """Reads the manifest.json's num_episodes field after ensuring episode 0
    has been read at least once (read_episode() writes the manifest the
    first time it runs, regardless of which episode is requested)."""
    manifest_path = Path("./export_manifest.json")
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)["num_episodes"]
    # Manifest doesn't exist yet; the first real read-episode call below
    # will create it. Caller handles this by reading episode 0 first.
    return -1


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--read-python", required=True, help="Path to the newer-lerobot venv's python.exe")
    parser.add_argument("--write-python", required=True, help="Path to this fork's own venv's python.exe")
    parser.add_argument("--episode-pickle", default="./export_episode_tmp.pkl")
    parser.add_argument("--manifest-path", default="./export_manifest.json")
    parser.add_argument("--local-root", default="./export_local_dataset")
    parser.add_argument("--state-path", default="./export_state.json")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--rewrite-task", default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    script = str(Path(__file__).parent / "export_dataset_to_v2_1.py")
    episode_pickle = Path(args.episode_pickle)
    manifest_path = Path(args.manifest_path)
    state_path = Path(args.state_path)

    # Load or initialize progress state.
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        print(f"Resuming from state file: {state['done_episodes']} episode(s) already done")
    else:
        state = {"done_episodes": []}

    # Read episode 0 first (with --episode-index=0) purely to populate the
    # manifest if it doesn't exist yet, so we know num_episodes. If episode
    # 0 was already read+written in a prior run, skip straight to reading
    # the manifest.
    if not manifest_path.exists():
        run([
            args.read_python, script, "--mode=read-episode",
            f"--source={args.source}", "--episode-index=0",
            f"--out-pickle={episode_pickle}", f"--manifest-path={manifest_path}",
            *(["--rewrite-task", args.rewrite_task] if args.rewrite_task else []),
        ])
        # This episode's pickle is now sitting on disk; it will be consumed
        # by the normal loop below (episode 0 is very likely not yet in
        # done_episodes on a fresh run).

    with open(manifest_path) as f:
        manifest = json.load(f)
    num_episodes = manifest["num_episodes"]
    if args.limit_episodes is not None:
        num_episodes = min(num_episodes, args.limit_episodes)

    print(f"Dataset has {num_episodes} episode(s) to export.")

    for ep_idx in range(num_episodes):
        if ep_idx in state["done_episodes"]:
            print(f"Episode {ep_idx} already done, skipping.")
            continue

        is_first = len(state["done_episodes"]) == 0
        is_last = ep_idx == num_episodes - 1

        # Read this episode, unless episode 0's pre-read above already
        # produced its pickle and it's still sitting there unconsumed.
        if not (ep_idx == 0 and episode_pickle.exists()):
            run([
                args.read_python, script, "--mode=read-episode",
                f"--source={args.source}", f"--episode-index={ep_idx}",
                f"--out-pickle={episode_pickle}", f"--manifest-path={manifest_path}",
                *(["--rewrite-task", args.rewrite_task] if args.rewrite_task else []),
            ])

        run([
            args.write_python, script, "--mode=write-episode",
            f"--out-pickle={episode_pickle}", f"--manifest-path={manifest_path}",
            f"--dest={args.dest}", f"--local-root={args.local_root}",
            *(["--first"] if is_first else []),
            *(["--last"] if is_last else []),
            *(["--private"] if args.private else []),
        ])

        episode_pickle.unlink(missing_ok=True)
        state["done_episodes"].append(ep_idx)
        with open(state_path, "w") as f:
            json.dump(state, f)

        print(f"Episode {ep_idx} done ({len(state['done_episodes'])}/{num_episodes} total).\n")

    print(f"\nAll {num_episodes} episodes exported and pushed to {args.dest}.")
    print(f"You can delete {args.local_root}, {manifest_path}, and {state_path} now.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        print("Re-run the same command to resume from the last completed episode.", file=sys.stderr)
        sys.exit(1)
