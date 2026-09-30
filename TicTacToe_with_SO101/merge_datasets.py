"""
Merge multiple existing LeRobotDataset repos (recorded separately, e.g. one
per grid cell) into a single new dataset repo, suitable for use as
`dataset.repo_id` in train_config.yaml.

Why this script exists: this lerobot version's training pipeline does not
support passing a list of dataset repo ids to `--dataset.repo_id` (both
`TrainPipelineConfig.validate()` and `make_dataset()` raise
NotImplementedError on a list). Training needs one single dataset, so any
data recorded across multiple separate repos has to be merged into one
before training.

IMPORTANT: run this script with a *newer* lerobot install (dataset format
v3.0+), not this repo's own forked lerobot (v0.2.0, dataset format v2.1).
Datasets recorded via LeLab or a freshly-installed lerobot are commonly
saved in format v3.0, which this fork's older code cannot read at all
(raises ForwardCompatibilityError). This script only touches raw dataset
data, not the act_lang policy code, so it is safe to run in a separate,
newer-lerobot virtual environment:

    python -m venv .merge_venv
    .merge_venv\\Scripts\\pip install "lerobot[dataset]"
    .merge_venv\\Scripts\\python merge_datasets.py --source ... --dest ...

Then switch back to this repo's own environment (the one with `pip install
-e .` already run) for the actual training step.

Usage:
    python merge_datasets.py \
        --source ai4y/tictactoe-position-1_20260929_232434 \
        --source ai4y/tictactoe-position-2_20260929_234618 \
        --source ai4y/tictactoe-position-3_20260930_002216 \
        --source ai4y/tictactoe-position-4_20260930_004253 \
        --source ai4y/tictactoe-position-5_20260930_010914 \
        --source ai4y/tictactoe-position-6_20260930_012153 \
        --source ai4y/tictactoe-position-7_20260930_014105 \
        --source ai4y/tictactoe-position-8_20260930_015641 \
        --source ai4y/tictactoe-position-9_20260930_022518 \
        --dest your-username/tictactoe

Test on a small, cheap slice first, before running the real merge:

    python merge_datasets.py \
        --source ai4y/tictactoe-position-1_20260929_232434 \
        --source ai4y/tictactoe-position-2_20260929_234618 \
        --dest your-username/tictactoe-merge-test \
        --limit-episodes-per-source 1

That copies just 1 episode from 2 source datasets into a throwaway test
repo, quick to run. Check the result on huggingface.co: it should show 2
episodes, and the images/video should look correct, not corrupted or
blank. If that looks right, delete the test repo and run the real merge
with all your real sources and no `--limit-episodes-per-source`.

This downloads every source dataset, re-encodes every frame's images into a
new dataset, and pushes the result once at the end. With 9 datasets of 10
episodes each (90 episodes total), this can take a while, most of it is
image/video re-encoding, not network time. Expect it to run for tens of
minutes, not seconds.

Requires `huggingface-cli login` to already be done (same as recording).

Task labels: the `task` string stored per frame in the merged dataset is
copied as-is from each source dataset (whatever `--dataset.single_task=`
was set to at record time for that source). This script does not rewrite
task strings. If your source datasets used a task string that does not
match what your game script sends at inference time (e.g. recorded as
"tictactoe-position-1" but the game sends "Place at Position 1"), training
will succeed but the trained policy will not respond to the game's actual
task strings. Check your source datasets' task strings against what your
game script sends before training on the merged result; use `--rewrite-task`
below to fix this during the merge if needed.
"""

import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def merge(
    source_repo_ids: list[str],
    dest_repo_id: str,
    private: bool = False,
    limit_episodes_per_source: int | None = None,
    rewrite_task: str | None = None,
) -> None:
    if not source_repo_ids:
        raise ValueError("Need at least one --source repo id.")

    print(f"Loading first source dataset to read its feature schema: {source_repo_ids[0]}")
    first = LeRobotDataset(source_repo_ids[0])

    # These are computed automatically by add_frame()/save_episode(); they must
    # not be passed in as data features when creating the new dataset, and
    # must not be copied from source frames either.
    auto_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}

    data_features = {k: v for k, v in first.features.items() if k not in auto_keys}
    image_keys = {k for k, v in data_features.items() if v.get("dtype") in ("video", "image")}

    dest = LeRobotDataset.create(
        repo_id=dest_repo_id,
        fps=first.fps,
        features=data_features,
        robot_type=first.meta.robot_type,
        use_videos=len(image_keys) > 0,
    )

    for source_repo_id in source_repo_ids:
        print(f"\nLoading source dataset: {source_repo_id}")
        source = LeRobotDataset(source_repo_id)

        source_data_keys = {k for k in source.features if k not in auto_keys}
        if source_data_keys != set(data_features.keys()):
            raise ValueError(
                f"{source_repo_id} has a different feature schema than "
                f"{source_repo_ids[0]}. All source datasets must have been "
                "recorded with the same --robot.cameras= keys and the same "
                "robot. Cannot merge datasets with different features."
            )

        episode_table = source.meta.episodes
        num_episodes = source.num_episodes
        if limit_episodes_per_source is not None:
            num_episodes = min(num_episodes, limit_episodes_per_source)

        for ep_idx in range(num_episodes):
            ep_row = episode_table[ep_idx]
            ep_start = ep_row["dataset_from_index"]
            ep_end = ep_row["dataset_to_index"]
            source_task = ep_row["tasks"][0] if ep_row["tasks"] else ""
            if rewrite_task == "auto-position":
                # Source tasks like "tictactoe-position-3" -> "Place at Position 3",
                # matching what play_TicTacToe.py sends at inference.
                digits = "".join(c for c in source_task if c.isdigit())
                if not digits:
                    raise ValueError(
                        f"--rewrite-task=auto-position could not find a "
                        f"position number in task string {source_task!r} "
                        f"from {source_repo_id}, episode {ep_idx}."
                    )
                task = f"Place at Position {digits}"
            elif rewrite_task is not None:
                task = rewrite_task
            else:
                task = source_task

            print(
                f"  Copying episode {ep_idx + 1}/{num_episodes} "
                f"from {source_repo_id} ({ep_end - ep_start} frames, task={task!r})"
            )

            for frame_idx in range(ep_start, ep_end):
                item = source[frame_idx]
                frame = {}
                for key in data_features:
                    value = item[key]
                    if key in image_keys:
                        # __getitem__ returns images as CHW float tensors;
                        # add_frame requires HWC.
                        value = value.permute(1, 2, 0)
                    frame[key] = value
                frame["task"] = task
                dest.add_frame(frame)

            dest.save_episode()

    dest.finalize()
    print(f"\nPushing merged dataset to the Hub: {dest_repo_id}")
    dest.push_to_hub(private=private)
    print(f"Done. {dest.num_episodes} episodes, {dest.num_frames} frames merged into {dest_repo_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        dest="sources",
        help="A source dataset repo id. Pass this flag once per dataset, in any order.",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="The new merged dataset's repo id, e.g. your-username/tictactoe",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push the merged dataset as a private repo (default: public).",
    )
    parser.add_argument(
        "--limit-episodes-per-source",
        type=int,
        default=None,
        help=(
            "Only copy this many episodes from each source dataset. Use a "
            "small number (e.g. 1) to sanity-check the script on a fast, "
            "cheap test run before doing the full merge."
        ),
    )
    parser.add_argument(
        "--rewrite-task",
        default=None,
        help=(
            "If set to 'auto-position', extracts the position number from "
            "each source episode's task string (e.g. 'tictactoe-position-3' "
            "or any string containing '3') and rewrites it to "
            "'Place at Position 3', matching what play_TicTacToe.py sends "
            "at inference. Use this if your source datasets were recorded "
            "with a task string that does not already match that format. "
            "If set to anything else, every copied frame's task string is "
            "replaced with that exact literal string instead (only "
            "correct for a single-task dataset, not a 9-cell one). Leave "
            "unset to keep each source dataset's original task string "
            "as-is."
        ),
    )
    args = parser.parse_args()
    merge(
        args.sources,
        args.dest,
        private=args.private,
        limit_episodes_per_source=args.limit_episodes_per_source,
        rewrite_task=args.rewrite_task,
    )
