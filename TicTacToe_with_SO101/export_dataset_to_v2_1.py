"""
Re-export a dataset from newer LeRobot's format (v3.0+) into this fork's
older format (v2.1), so this fork's own training code (and the custom
act_lang policy, which only exists in this fork) can actually read it.

Why this script exists: this fork is pinned to lerobot 0.2.0
(CODEBASE_VERSION = "v2.1" in lerobot/datasets/lerobot_dataset.py). A
dataset recorded with a newer lerobot install (for example, via LeLab, or
any lerobot installed after this fork was branched) is saved in a newer
format (v3.0+ as of this writing) that this fork's dataset-loading code
cannot parse at all; it raises ForwardCompatibilityError immediately.
Porting the custom act_lang policy to run under newer lerobot instead was
evaluated and rejected for now: newer lerobot moved normalization out of
the policy class entirely into an external processor pipeline, among other
structural differences, making that a much larger, multi-hour engineering
task, not a quick fix. Re-exporting the dataset the other direction (new
format -> old format this fork already reads) is the smaller, faster,
already-proven-pattern fix.

IMPORTANT: run this script with a *newer* lerobot install (dataset format
v3.0+) to do the reading half, not this repo's own forked lerobot. This
mirrors merge_datasets.py exactly:

    python -m venv .export_venv
    .export_venv\\Scripts\\pip install "lerobot[dataset]"
    .export_venv\\Scripts\\python export_dataset_to_v2_1.py --source ... --dest ...

The read side (LeRobotDataset(repo_id), __getitem__, meta.episodes with
dataset_from_index/dataset_to_index) only works with a newer lerobot
install. The WRITE side (LeRobotDataset.create(), add_frame(frame,
task=...), CHW image tensors accepted directly) is this fork's own older
API and only exists in this fork's own installed lerobot (the one from
`pip install -e .` in TicTacToe_with_SO101/). This script imports
`lerobot.datasets.lerobot_dataset` at call time from whichever environment
actually runs it; it cannot be run once against both formats in a single
environment, because only one lerobot version can be installed at a time.
The practical way to run it, given that constraint: run it with the newer
lerobot install for reading (as shown above), which pulls the exported
episode dict data as plain Python/numpy/torch objects, but writing still
needs this fork's older `LeRobotDataset.create()` API to produce a v2.1
dataset. See "Two-environment execution" below for the actual mechanism
used to bridge this.

Usage:
    python export_dataset_to_v2_1.py \
        --source IndiaTechTeamSL2/tictactoe \
        --dest IndiaTechTeamSL2/tictactoe-v2_1

Test on a small slice first:
    python export_dataset_to_v2_1.py \
        --source IndiaTechTeamSL2/tictactoe \
        --dest IndiaTechTeamSL2/tictactoe-v2_1-test \
        --limit-episodes 2

After exporting, point `train_config.yaml`'s `dataset.repo_id` at the new
`--dest` repo, which this fork's own lerobot (v0.2.0 / v2.1) can actually
load.

## Two-environment execution

This script's `read_source()` function requires the newer lerobot's
`LeRobotDataset` read API (dataset_from_index/dataset_to_index,
__getitem__ decoding video frames, meta.episodes with the `tasks` column).
This fork's older `LeRobotDataset` cannot read the v3.0+ source dataset at
all, so `read_source()` cannot run inside this fork's own venv.

This script's `write_dest()` function requires this fork's `add_frame(frame,
task=...)` two-argument call signature and `LeRobotDataset.create()`'s
older parameter set. Newer lerobot's write API has also changed (per the
same investigation that ruled out porting act_lang), so `write_dest()`
cannot run inside the newer lerobot venv either.

Because of this, `main()` below does NOT try to do both in one process.
Instead: run this script once with `--mode=read`, using the newer-lerobot
venv, which reads the full source dataset into memory and pickles it to a
local file. Then run this script again with `--mode=write`, using this
fork's own venv (the one `pip install -e .` set up in
`TicTacToe_with_SO101/`), which loads that pickle and writes + pushes the
v2.1 dataset. `--mode=full` (the default) is only valid if somehow both
APIs are importable in the same environment, which they are not for these
two lerobot versions; use `--mode=read` then `--mode=write` in practice.

    # Step 1, in the newer-lerobot venv:
    python export_dataset_to_v2_1.py --mode=read \
        --source IndiaTechTeamSL2/tictactoe \
        --pickle-path ./export_data.pkl \
        --limit-episodes 2   # drop this for the real, full export

    # Step 2, in this fork's own venv (TicTacToe_with_SO101's pip install -e .):
    python export_dataset_to_v2_1.py --mode=write \
        --pickle-path ./export_data.pkl \
        --dest IndiaTechTeamSL2/tictactoe-v2_1-test
"""

import argparse
import pickle


def read_source(source_repo_id: str, limit_episodes: int | None) -> dict:
    """Run this in a newer-lerobot environment. Returns a plain-Python dict
    with everything write_dest() needs, no lerobot objects inside it."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading source dataset: {source_repo_id}")
    source = LeRobotDataset(source_repo_id)

    auto_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    data_keys = [k for k in source.features if k not in auto_keys]
    image_keys = {k for k in data_keys if source.features[k].get("dtype") in ("video", "image")}

    features_out = {}
    for k in data_keys:
        f = source.features[k]
        features_out[k] = {"dtype": f["dtype"], "shape": f["shape"], "names": f.get("names")}

    episode_table = source.meta.episodes
    num_episodes = source.num_episodes
    if limit_episodes is not None:
        num_episodes = min(num_episodes, limit_episodes)

    episodes_out = []
    for ep_idx in range(num_episodes):
        ep_row = episode_table[ep_idx]
        ep_start = ep_row["dataset_from_index"]
        ep_end = ep_row["dataset_to_index"]
        task = ep_row["tasks"][0] if ep_row["tasks"] else ""

        print(f"  Reading episode {ep_idx + 1}/{num_episodes} ({ep_end - ep_start} frames, task={task!r})")

        frames = []
        for frame_idx in range(ep_start, ep_end):
            item = source[frame_idx]
            frame = {}
            for k in data_keys:
                v = item[k]
                if k in image_keys:
                    # CHW float tensor -> HWC numpy uint8, plain data, no torch
                    # object needs to survive the pickle/env boundary as a tensor.
                    v = (v.permute(1, 2, 0).clamp(0, 1) * 255).to("cpu").numpy().astype("uint8")
                else:
                    v = v.cpu().numpy() if hasattr(v, "cpu") else v
                frame[k] = v
            frames.append(frame)

        episodes_out.append({"task": task, "frames": frames})

    return {
        "fps": source.fps,
        "robot_type": source.meta.robot_type,
        "features": features_out,
        "image_keys": image_keys,
        "episodes": episodes_out,
    }


def write_dest(data: dict, dest_repo_id: str, private: bool = False) -> None:
    """Run this in THIS FORK's own environment (v0.2.0 / v2.1 lerobot)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {}
    for k, f in data["features"].items():
        features[k] = {"dtype": f["dtype"], "shape": f["shape"], "names": f["names"]}

    dest = LeRobotDataset.create(
        repo_id=dest_repo_id,
        fps=data["fps"],
        features=features,
        robot_type=data["robot_type"],
        use_videos=len(data["image_keys"]) > 0,
    )

    for i, ep in enumerate(data["episodes"]):
        print(f"  Writing episode {i + 1}/{len(data['episodes'])} (task={ep['task']!r})")
        for frame in ep["frames"]:
            dest.add_frame(frame, task=ep["task"])
        dest.save_episode()

    print(f"\nPushing re-exported dataset to the Hub: {dest_repo_id}")
    dest.push_to_hub(private=private)
    print(f"Done. {dest.num_episodes} episodes merged into {dest_repo_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["read", "write"], required=True)
    parser.add_argument("--source", help="Source dataset repo id (--mode=read only).")
    parser.add_argument("--dest", help="Destination dataset repo id (--mode=write only).")
    parser.add_argument("--pickle-path", default="./export_data.pkl")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if args.mode == "read":
        if not args.source:
            raise ValueError("--source is required for --mode=read")
        data = read_source(args.source, args.limit_episodes)
        with open(args.pickle_path, "wb") as f:
            pickle.dump(data, f)
        print(f"\nWrote {len(data['episodes'])} episodes to {args.pickle_path}")
        print(f"Now run --mode=write in this fork's own venv to finish the export.")
    else:
        if not args.dest:
            raise ValueError("--dest is required for --mode=write")
        with open(args.pickle_path, "rb") as f:
            data = pickle.load(f)
        write_dest(data, args.dest, private=args.private)
