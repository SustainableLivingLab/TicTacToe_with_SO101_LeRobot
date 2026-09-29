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

This script has not been run against real data before being handed to you.
Test it on a small, cheap slice first, before running the real merge:

    python merge_datasets.py \
        --source ai4y/tictactoe-position-1_20260929_232434 \
        --source ai4y/tictactoe-position-2_20260929_234618 \
        --dest ai4y/tictactoe-merge-test \
        --limit-episodes-per-source 1

That copies just 1 episode from 2 source datasets into a throwaway test
repo, quick to run. Check the result at
https://huggingface.co/datasets/ai4y/tictactoe-merge-test : it should show
2 episodes, and the images/video should look correct, not corrupted or
blank. If that looks right, delete the test repo and run the real merge
with all 9 sources and no `--limit-episodes-per-source`:

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
        --dest ai4y/tictactoe

This downloads every source dataset, re-encodes every frame's images into a
new dataset, and pushes the result once at the end. With 9 datasets of 10
episodes each (90 episodes total), this can take a while, most of it is
image/video re-encoding, not network time. Expect it to run for tens of
minutes, not seconds.

Requires `huggingface-cli login` to already be done (same as recording).
"""

import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def merge(
    source_repo_ids: list[str],
    dest_repo_id: str,
    private: bool = False,
    limit_episodes_per_source: int | None = None,
) -> None:
    if not source_repo_ids:
        raise ValueError("Need at least one --source repo id.")

    print(f"Loading first source dataset to read its feature schema: {source_repo_ids[0]}")
    first = LeRobotDataset(source_repo_ids[0])

    dest = LeRobotDataset.create(
        repo_id=dest_repo_id,
        fps=first.fps,
        features=first.features,
        robot_type=first.meta.robot_type,
        use_videos=len(first.meta.video_keys) > 0,
    )

    for source_repo_id in source_repo_ids:
        print(f"\nLoading source dataset: {source_repo_id}")
        source = LeRobotDataset(source_repo_id)

        if set(source.features.keys()) != set(dest.features.keys()):
            raise ValueError(
                f"{source_repo_id} has a different feature schema than "
                f"{source_repo_ids[0]}. All source datasets must have been "
                "recorded with the same --robot.cameras= keys and the same "
                "robot. Cannot merge datasets with different features."
            )

        num_episodes = source.num_episodes
        if limit_episodes_per_source is not None:
            num_episodes = min(num_episodes, limit_episodes_per_source)

        for ep_idx in range(num_episodes):
            ep_start = source.episode_data_index["from"][ep_idx].item()
            ep_end = source.episode_data_index["to"][ep_idx].item()
            print(
                f"  Copying episode {ep_idx + 1}/{source.num_episodes} "
                f"from {source_repo_id} ({ep_end - ep_start} frames)"
            )

            for frame_idx in range(ep_start, ep_end):
                item = source[frame_idx]
                task = item.pop("task")
                for key in ("episode_index", "frame_index", "index", "timestamp", "task_index"):
                    item.pop(key, None)
                dest.add_frame(item, task=task)

            dest.save_episode()

    print(f"\nPushing merged dataset to the Hub: {dest_repo_id}")
    dest.push_to_hub(private=private)
    print(f"Done. {dest.num_episodes} episodes, {dest.num_frames} frames merged into {dest_repo_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()
    merge(
        args.sources,
        args.dest,
        private=args.private,
        limit_episodes_per_source=args.limit_episodes_per_source,
    )
