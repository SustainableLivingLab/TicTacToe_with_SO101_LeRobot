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
evaluated and rejected: newer lerobot's PreTrainedPolicy contract changed
structurally (predict_action_chunk/select_action split, get_optim_params
instead of get_optimizer_preset, normalization moved out of the policy
entirely into an external processor pipeline applied by the training
loop). That is genuine multi-hour correctness-critical engineering, not a
quick fix. Re-exporting the dataset the other direction (new format -> old
format this fork already reads) is the smaller, faster fix.

IMPORTANT: run this script with a *newer* lerobot install (dataset format
v3.0+) to do the reading half, not this repo's own forked lerobot:

    python -m venv .export_venv_read
    .export_venv_read\\Scripts\\pip install "lerobot[dataset]"

And this fork's own venv (this repo's `pip install -e .`) for the write
half. The two lerobot versions cannot be installed in the same
environment, so no single Python process can do both; see "Per-episode
interleaving" below for how the two are combined without ever staging the
whole dataset on disk at once.

## Per-episode interleaving (important: do not stage the whole dataset)

Two earlier versions of this script staged the whole dataset before doing
anything else, first entirely in memory (grew past 18GB and climbing for
this project's own 90-episode dataset), then as ~164GB of raw uncompressed
pickle files on disk (all 90 episodes' raw frames at once), which does not
fit on any drive on the machine this was built on. This version processes
one episode fully (read from source, write into the growing local v2.1
dataset, delete the intermediate pickle) before moving to the next, so at
most one episode's raw pickle (a few GB) exists on disk at any time, on
top of the growing but already video-encoded (much smaller) destination
dataset. Only the very last episode triggers a push to the Hub.

This script itself only implements the single-episode read and write
primitives (`--mode=read-episode`, `--mode=write-episode`), each callable
from its own venv. `export_orchestrate.py` (in this repo, plain Python, no
lerobot import, runs in whichever Python is convenient) alternates between
the two venvs' interpreters via subprocess, one episode at a time, and is
the script you actually run end to end. Read that script's own docstring
for the real usage; the two `--mode`s here exist mainly for the
orchestrator to call, though they also work run by hand for one episode at
a time if you want to drive this manually instead.

If your dataset's task strings do not already match the
`"Place at Position N"` format `play_TicTacToe.py` sends at inference,
pass `--rewrite-task auto-position` on `--mode=read-episode` to fix them
during the export, same as `merge_datasets.py`.

If the source dataset is not tagged with a Hub version ref matching its
`meta/info.json` `codebase_version`, reading fails immediately with
`RuntimeError: Your dataset must be tagged with a codebase version.` Tag
it once, using the exact version from that file, before retrying:

    python -c "
    from huggingface_hub import hf_hub_download, HfApi
    import json
    path = hf_hub_download('<repo_id>', 'meta/info.json', repo_type='dataset')
    version = json.load(open(path))['codebase_version']
    HfApi().create_tag('<repo_id>', tag=version, repo_type='dataset')
    "
"""

import argparse
import json
import pickle
from pathlib import Path


def read_episode(
    source_repo_id: str,
    episode_index: int,
    out_pickle_path: Path,
    manifest_path: Path,
    rewrite_task: str | None,
) -> None:
    """Run this in a newer-lerobot environment. Reads exactly one episode
    and writes it (plus a manifest.json, only if it doesn't already exist)
    to disk. Caller is responsible for deleting out_pickle_path once it has
    been consumed by write_episode()."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading source dataset metadata: {source_repo_id}")
    source = LeRobotDataset(source_repo_id)

    auto_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    data_keys = [k for k in source.features if k not in auto_keys]
    image_keys = sorted(k for k in data_keys if source.features[k].get("dtype") in ("video", "image"))

    if not manifest_path.exists():
        features_out = {}
        for k in data_keys:
            f = source.features[k]
            features_out[k] = {"dtype": f["dtype"], "shape": f["shape"], "names": f.get("names")}
        manifest = {
            "fps": source.fps,
            "robot_type": source.meta.robot_type,
            "num_episodes": source.num_episodes,
            "features": features_out,
            "image_keys": image_keys,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

    episode_table = source.meta.episodes
    ep_row = episode_table[episode_index]
    ep_start = ep_row["dataset_from_index"]
    ep_end = ep_row["dataset_to_index"]
    source_task = ep_row["tasks"][0] if ep_row["tasks"] else ""

    if rewrite_task == "auto-position":
        digits = "".join(c for c in source_task if c.isdigit())
        if not digits:
            raise ValueError(
                f"--rewrite-task=auto-position could not find a position number "
                f"in task string {source_task!r}, episode {episode_index}."
            )
        task = f"Place at Position {digits}"
    elif rewrite_task is not None:
        task = rewrite_task
    else:
        task = source_task

    print(
        f"  Reading episode {episode_index} ({ep_end - ep_start} frames, "
        f"task={task!r}) -> {out_pickle_path}"
    )

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

    out_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_pickle_path.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump({"task": task, "frames": frames}, f)
    tmp_path.rename(out_pickle_path)  # atomic-ish: no half-written file left behind


def write_episode(
    episode_pickle_path: Path,
    manifest_path: Path,
    dest_repo_id: str,
    local_root: Path,
    is_first_episode: bool,
    is_last_episode: bool,
    private: bool = False,
) -> None:
    """Run this in THIS FORK's own environment (v0.2.0 / v2.1 lerobot).
    Adds exactly one episode to the local dataset at local_root, creating
    it fresh if is_first_episode, otherwise resuming the existing one.
    Pushes to the Hub only if is_last_episode."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with open(manifest_path) as f:
        manifest = json.load(f)

    with open(episode_pickle_path, "rb") as f:
        ep = pickle.load(f)

    if is_first_episode:
        features = {}
        for k, f in manifest["features"].items():
            # JSON round-trip turns tuples into lists; this fork's
            # validate_frame() does an exact `value.shape != expected_shape`
            # comparison, and np.ndarray.shape is always a tuple, so
            # expected_shape must be a tuple too or every add_frame() call
            # fails validation.
            features[k] = {"dtype": f["dtype"], "shape": tuple(f["shape"]), "names": f["names"]}

        dest = LeRobotDataset.create(
            repo_id=dest_repo_id,
            fps=manifest["fps"],
            features=features,
            root=local_root,
            robot_type=manifest["robot_type"],
            use_videos=len(manifest["image_keys"]) > 0,
        )
    else:
        dest = LeRobotDataset(dest_repo_id, root=local_root)

    print(f"  Writing episode from {episode_pickle_path.name} (task={ep['task']!r}) into {local_root}")
    for frame in ep["frames"]:
        dest.add_frame(frame, task=ep["task"])
    dest.save_episode()

    if is_last_episode:
        print(f"\nPushing re-exported dataset to the Hub: {dest_repo_id}")
        dest.push_to_hub(private=private)
        print(f"Done. {dest.num_episodes} episodes pushed to {dest_repo_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["read-episode", "write-episode"], required=True)

    # read-episode args
    parser.add_argument("--source", help="Source dataset repo id (--mode=read-episode).")
    parser.add_argument("--episode-index", type=int, help="Which episode to process (both modes).")
    parser.add_argument("--out-pickle", help="Where to write/read the one-episode pickle (both modes).")
    parser.add_argument(
        "--manifest-path", default="./export_manifest.json", help="Shared manifest file (both modes)."
    )
    parser.add_argument("--rewrite-task", default=None, help="See module docstring.")

    # write-episode args
    parser.add_argument("--dest", help="Destination dataset repo id (--mode=write-episode).")
    parser.add_argument("--local-root", help="Local dataset directory (--mode=write-episode).")
    parser.add_argument("--first", action="store_true", help="This is episode 0: create the dataset.")
    parser.add_argument("--last", action="store_true", help="This is the final episode: push to Hub.")
    parser.add_argument("--private", action="store_true")

    args = parser.parse_args()

    if args.mode == "read-episode":
        if not (args.source and args.episode_index is not None and args.out_pickle):
            raise ValueError("--source, --episode-index, --out-pickle are required for --mode=read-episode")
        read_episode(
            args.source,
            args.episode_index,
            Path(args.out_pickle),
            Path(args.manifest_path),
            args.rewrite_task,
        )
    else:
        if not (args.dest and args.local_root and args.out_pickle):
            raise ValueError("--dest, --local-root, --out-pickle are required for --mode=write-episode")
        write_episode(
            Path(args.out_pickle),
            Path(args.manifest_path),
            args.dest,
            Path(args.local_root),
            is_first_episode=args.first,
            is_last_episode=args.last,
            private=args.private,
        )
