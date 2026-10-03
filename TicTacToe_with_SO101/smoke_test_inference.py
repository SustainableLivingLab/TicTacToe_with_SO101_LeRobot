"""
Smoke test for the trained act_lang checkpoint, no physical robot needed.

Builds the same observation/action feature schema play_TicTacToe.py builds
from a real SO101Follower (6 motor positions, front/top cameras), but with
synthetic data instead of a live robot connection. Loads the checkpoint from
the Hub, runs predict_action() once, confirms the model returns an action
tensor of the right shape. Does not test the camera/Gemini/robot pipeline,
only confirms the checkpoint itself loads and runs inference correctly.

Usage:
    python smoke_test_inference.py --policy.path=IndiaTechTeamSL2/tictactoe-act-lang
    python smoke_test_inference.py --policy.path=IndiaTechTeamSL2/tictactoe-act-lang --task="Place at Position 3"
"""

import numpy as np

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs import parser
from lerobot.configs.parser import wrap
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import make_policy
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from dataclasses import dataclass


MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class MockDatasetMetadata:
    def __init__(self, features: dict):
        self.features = features
        self.stats = {}


@dataclass
class SmokeTestConfig:
    policy: PreTrainedConfig | None = None
    task: str = "Place at Position 5"

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        if self.policy is None:
            raise ValueError("Pass --policy.path=<repo_id or local dir>")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


@wrap()
def main(cfg: SmokeTestConfig):
    if cfg.policy is None:
        raise ValueError("Pass --policy.path=<repo_id or local dir>")

    action_features_raw = {f"{m}.pos": float for m in MOTOR_NAMES}
    observation_features_raw = {
        **action_features_raw,
        "front": (480, 640, 3),
        "top": (480, 640, 3),
    }

    action_features = hw_to_dataset_features(action_features_raw, "action", use_video=True)
    obs_features = hw_to_dataset_features(observation_features_raw, "observation", use_video=True)
    dataset_features = {**action_features, **obs_features}
    metadata = MockDatasetMetadata(dataset_features)

    print(f"Loading policy from {cfg.policy.pretrained_path}...")
    policy = make_policy(cfg.policy, ds_meta=metadata)
    policy.reset()
    print("Policy loaded OK.")

    observation = {
        **{f"{m}.pos": 0.0 for m in MOTOR_NAMES},
        "front": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "top": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    }
    observation_frame = build_dataset_frame(metadata.features, observation, prefix="observation")

    print("Running predict_action() with synthetic observation...")
    action = predict_action(
        observation_frame,
        policy,
        get_safe_torch_device(policy.config.device),
        policy.config.use_amp,
        task=cfg.task,
        robot_type="so101_follower",
    )

    print(f"Action shape: {tuple(action.shape)}")
    print(f"Action values: {action.tolist()}")
    assert action.shape[0] == len(MOTOR_NAMES), f"Expected {len(MOTOR_NAMES)} action dims, got {action.shape[0]}"
    print("\nSMOKE TEST PASSED: checkpoint loads and produces a correctly-shaped action.")


if __name__ == "__main__":
    main()
