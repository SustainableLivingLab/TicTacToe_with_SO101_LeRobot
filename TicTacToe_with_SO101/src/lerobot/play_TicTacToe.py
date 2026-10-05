import cv2
from PIL import Image
import numpy as np
import io
import os
import random
import time
from google import genai
from google.genai import types
from pathlib import Path
from dataclasses import dataclass
from lerobot.datasets.utils import (
    build_dataset_frame,
    hw_to_dataset_features,
    DEFAULT_FEATURES
    )
from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
import lerobot.cameras.opencv.configuration_opencv  # noqa: F401
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
import lerobot.robots.so101_follower.config_so101_follower  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs import parser
from lerobot.policies.factory import make_policy

from lerobot.utils.control_utils import (
    init_keyboard_listener,
    predict_action,
)
from lerobot.utils.utils import (
    get_safe_torch_device,
    log_say,
)
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.utils.robot_utils import busy_wait
from contextlib import contextmanager
from typing import Optional
import re

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable not set. "
        "Set it in your shell or in a local .env file (see .env.example)."
    )
client = genai.Client(api_key=GEMINI_API_KEY)

# Physical piece colors. Human always plays X, robot always plays O.
# Override per physical board via .env, e.g. X_COLOR=Black for a
# Black/Red piece set instead of the default Blue/Red set.
X_COLOR = os.environ.get("X_COLOR", "Blue")
O_COLOR = os.environ.get("O_COLOR", "Red")

class MockDatasetMetadata:
    """Mock metadata object to satisfy make_policy requirements"""
    def __init__(self, features: dict, stats: dict = None):
        self.features = features
        self.stats = stats or {}

@dataclass
class TicTacToeConfig:
    robot: RobotConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for the robot to play its turn
    robot_turn_time_s: int | float = 30
    # Number of seconds for the human player to play their turn
    player_turn_time_s: int | float = 10
    # OpenCV index of the separate camera Gemini reads the board from (not the
    # robot's front/top cameras). Camera numbers can change after replugging;
    # check with `python -m lerobot.find_cameras opencv`.
    board_camera_index: int = 2
    # Encode frames in the dataset into video
    use_videos: bool = True
    policy: PreTrainedConfig | None = None
    # Metadata for policy
    metadata: MockDatasetMetadata | None = None


    revision: str | None = None
    force_cache_sync: bool = False

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("Choose a policy")
        
        robot = make_robot_from_config(self.robot)
        self.metadata = create_mock_metadata(robot, self.use_videos)

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@contextmanager
def robot_context(cfg: TicTacToeConfig):
    """Context manager for robot connection."""
    robot = make_robot_from_config(cfg.robot)
    listener = None
    try:
        robot.connect()
        listener, events = init_keyboard_listener()
        yield robot, events
    finally:
        robot.disconnect()
        if listener:
            listener.stop()

def create_mock_metadata(robot, use_videos: bool = True) -> MockDatasetMetadata:
    """Create minimal metadata needed for make_policy"""
    action_features = hw_to_dataset_features(robot.action_features, "action", use_videos)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_videos)
    dataset_features = {**action_features, **obs_features}
    
    # Combine with default features (same as LeRobotDatasetMetadata.create())
    features = {**dataset_features, **DEFAULT_FEATURES}
    
    # Empty stats (same as new dataset)
    stats = {}
    
    return MockDatasetMetadata(features, stats)

def call_policy(cfg: TicTacToeConfig, instruction: str, policy=None):
    """Execute policy with proper resource management.

    `policy` is loaded once in play() and passed in on every turn, rather
    than reloaded here each call, since make_policy() re-downloading and
    rebuilding the checkpoint added avoidable delay to every single robot
    turn. Falls back to loading it locally if not given, so this function
    still works standalone."""
    with robot_context(cfg) as (robot, events):

        if not check_pickup_zone_has_piece(robot):
            print("No piece detected in the pickup zone. Waiting for it to be refilled.")
            log_say("Please refill the pickup spot", cfg.play_sounds)
            while not check_pickup_zone_has_piece(robot):
                if events["exit_early"]:
                    events["exit_early"] = False
                    return
                busy_wait(2)

        if policy is None:
            policy = make_policy(cfg.policy, ds_meta=cfg.metadata)

        matches = re.findall(r'Place at position \d+', instruction, re.IGNORECASE)
        if not matches:
            raise ValueError(
                f"Instruction {instruction!r} does not contain a 'Place at position N' directive."
            )
        instruction = matches[-1]

        if policy is not None:
            policy.reset()

        # Settling check: once the policy's predicted action stops changing
        # meaningfully between consecutive steps, the trained trajectory has
        # finished and it is just holding the final pose. There is no true
        # completion signal (no force sensor, no vision check, ACT does not
        # predict a stop token), so this is a heuristic, not a guarantee.
        # min_settle_check_s gives the arm time to actually start moving
        # before the check can trigger, so an initially-still pose at the
        # start of the turn is never mistaken for "done".
        settle_tolerance = 1.0  # degrees/units, per joint, between consecutive actions
        settle_frames_required = 10  # consecutive stable frames before declaring done
        min_settle_check_s = 3.0
        stable_frame_count = 0
        previous_action = None

        timestamp = 0
        start_episode_t = time.perf_counter()

        while timestamp < cfg.robot_turn_time_s:
            start_loop_t = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                break

            observation = robot.get_observation()

            if policy is not None:
                observation_frame = build_dataset_frame(cfg.metadata.features, observation, prefix="observation")
                action_values = predict_action(
                    observation_frame,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    policy.config.use_amp,
                    task=instruction,
                    robot_type=robot.robot_type,
                )
                action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
                robot.send_action(action)

                if timestamp >= min_settle_check_s:
                    if previous_action is not None and all(
                        abs(action[k] - previous_action[k]) < settle_tolerance for k in action
                    ):
                        stable_frame_count += 1
                        if stable_frame_count >= settle_frames_required:
                            break
                    else:
                        stable_frame_count = 0
                previous_action = action

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / cfg.fps - dt_s)
            timestamp = time.perf_counter() - start_episode_t


def crop_image(image, left_pct, right_pct, top_pct, bottom_pct):
    """
    Crop the image
    """
    width, height = image.size
    
    left = int(width * left_pct)    # Start from about x% from left
    right = int(width * right_pct)   # End at about x% from left
    top = int(height * top_pct)    # Start from about x% from top
    bottom = int(height * bottom_pct) # End at about x% from top
    
    image = image.crop((left, top, right, bottom))

    return image

def get_grid_image(camera_index: int) -> Optional[Image.Image]:
    """Capture image from the camera."""

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open board camera at index {camera_index} (in use by another app, or the index changed).")
    try:
        cap.set(3, 640)
        cap.set(4, 480)
        
        # Warm up camera
        for _ in range(5):
            cap.read()
        
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
        else:
            print(f"Error capturing image")
            image = None
    finally:
        cap.release()
    
    return image

def transform_to_top_view(pil_image, four_points, output_size=None):
    """
    Transform a PIL image from front view to top view using perspective transformation
    
    Args:
        pil_image: PIL Image object (input image)
        four_points: List of 4 coordinate tuples in anti-clockwise order from bottom-left
                    [(bottom_left_x, bottom_left_y), (bottom_right_x, bottom_right_y), 
                     (top_right_x, top_right_y), (top_left_x, top_left_y)]
        output_size: Optional tuple (width, height) for output image size
                    If None, uses original image dimensions
    
    Returns:
        PIL Image: Transformed image showing top-down view
    """
    
    # Convert PIL image to OpenCV format
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    height, width = cv_image.shape[:2]
    
    # Set output size
    if output_size is None:
        output_width, output_height = width, height
    else:
        output_width, output_height = output_size
    
    # Extract points in anti-clockwise order from bottom-left
    bottom_left = four_points[0]
    bottom_right = four_points[1]
    top_right = four_points[2]
    top_left = four_points[3]
    
    # Source points (the quadrilateral in the original image)
    # OpenCV expects points in order: top-left, top-right, bottom-right, bottom-left
    src_points = np.float32([
        top_left,      # Top-left
        top_right,     # Top-right
        bottom_right,  # Bottom-right
        bottom_left    # Bottom-left
    ])
    
    # Destination points (perfect rectangle for top-down view)
    # Add some padding to avoid edge artifacts
    padding = 20
    dst_points = np.float32([
        [padding, padding],                                    # Top-left
        [output_width - padding, padding],                     # Top-right
        [output_width - padding, output_height - padding],     # Bottom-right
        [padding, output_height - padding]                     # Bottom-left
    ])
    
    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply the perspective transformation
    warped = cv2.warpPerspective(cv_image, matrix, (output_width, output_height))
    
    # Convert back to PIL format
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(warped_rgb)


def process_images_with_LLM(image: Image.Image, prompt: str) -> Optional[str]:
    """Call the LLM API with the image and prompt"""

    contents = []

    if image is not None:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        
        contents.append(types.Part.from_bytes(
            data=img_bytes,
            mime_type='image/jpeg',
        ))
    
    # Add the prompt
    contents.append(prompt)
    
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=contents,
        config={
            "temperature": 0.0
        }
    )
    
    return response

def get_LLM_output(image: Image.Image) -> str:
    """Get LLM decision for next move."""
    prompt = f""""
            The attached images show a 3x3 grid board used for playing the game with tokens.
            The image is the whole camera view, so it also shows the table and other objects;
            read only the 3x3 grid board.

            The board orientation is as follows:

            Top Row:
            Position 1 | Position 2 | Position 3
            Middle Row:
            Position 4 | Position 5 | Position 6
            Bottom Row:
            Position 7 | Position 8 | Position 9

            1 | 2 | 3
            ---------
            4 | 5 | 6
            ---------
            7 | 8 | 9

            Mention the state of the board in the following format:

            Position 1: Empty/{O_COLOR}/{X_COLOR}
            Position 2: Empty/{O_COLOR}/{X_COLOR}
            And so on

            """

    response = process_images_with_LLM(image, prompt)
    output_string = response.text

    # print(output_string)

    return output_string

def check_pickup_zone_has_piece(robot) -> bool:
    """Ask Gemini whether a loose piece is visible and ready to be picked up,
    using the already-connected robot's own front camera (the same frame the
    ACT policy itself observes). No hardcoded crop or camera index: this
    reuses robot.get_observation(), so it stays correct even if the camera
    is repositioned, unlike the fixed-crop board-reading pipeline."""
    observation = robot.get_observation()
    frame = observation.get("front")
    if frame is None:
        print("No 'front' camera observation available; treating pickup zone as empty.")
        return False  # fail closed: never start the robot's turn on missing/uncertain data

    image = Image.fromarray(frame)

    prompt = f""""
            This image is a live camera frame from a robot arm's workspace,
            used to play a tile-placing game. There is a fixed pickup spot
            somewhere in this frame where a single loose {O_COLOR} game piece
            is placed before each of the robot's turns, for the robot to
            grab and move onto a board.

            Look carefully at the full image and decide: is there currently
            a loose, ungrasped {O_COLOR} piece sitting in that pickup spot,
            fully visible and ready for the robot's gripper to pick up right
            now?

            Answer NO if: the pickup spot is empty, you cannot clearly see
            the pickup spot, the piece is already inside the robot's gripper
            or being held/moved, you only see pieces already placed on the
            board, or you are not confident a loose piece is there.

            Answer YES only if you can clearly see one loose {O_COLOR} piece
            sitting by itself, not touching the gripper, at the pickup spot.

            Respond with exactly one word: YES or NO. No other text.
            """

    response = process_images_with_LLM(image, prompt)
    answer = (response.text or "").strip().upper()
    print(f"Pickup-zone check, Gemini says: {answer[:40]!r}")
    return answer.startswith("YES")

class BoardReadError(ValueError):
    """Gemini's board reply could not be fully understood; `partial` holds
    what was understood (None for positions that were missing/ambiguous)."""

    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


def parse_board_state(board_string):
    """
    Parse Gemini's board description into a vector of size 9.

    Returns:
        list: Vector where -1 = X_COLOR, 1 = O_COLOR, 0 = Empty
              (X_COLOR/O_COLOR set via env vars, default Blue/Red)

    Raises:
        ValueError: if any of the 9 positions is missing or its state is not
        exactly one of empty / X_COLOR / O_COLOR. An unrecognised cell used to
        be silently read as empty, which made the robot choose cells that
        already held a piece; refusing to guess is safer.
    """
    x_word = X_COLOR.strip().lower()
    o_word = O_COLOR.strip().lower()
    vector = [None] * 9
    problems = []

    for raw_line in board_string.splitlines():
        line = raw_line.replace("*", " ").replace("`", " ").strip(" -	")
        m = re.search(r"position\s*(\d)\s*[:=\-]\s*(.+)", line, re.IGNORECASE)
        if not m:
            continue
        index = int(m.group(1)) - 1
        if not 0 <= index <= 8:
            problems.append(f"position out of range: {raw_line.strip()!r}")
            continue
        words = set(re.findall(r"[a-z]+", m.group(2).lower()))
        matches = [value for word, value in (("empty", 0), (x_word, -1), (o_word, 1)) if word in words]
        if len(matches) != 1:
            problems.append(f"cannot tell the state of position {index + 1}: {raw_line.strip()!r}")
            continue
        vector[index] = matches[0]

    missing = [i + 1 for i, v in enumerate(vector) if v is None]
    if missing or problems:
        raise BoardReadError(
            f"Could not read the whole board (missing positions: {missing}; issues: {problems}). "
            f"Expected each line as 'Position N: Empty/{X_COLOR}/{O_COLOR}'.",
            vector,
        )
    return vector

def analyzeboard(board):
    cb=[[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]

    for i in range(0,8):
        if(board[cb[i][0]] != 0 and
           board[cb[i][0]] == board[cb[i][1]] and
           board[cb[i][0]] == board[cb[i][2]]):
            return board[cb[i][2]]
    return 0

def minimax(board,player):
    x=analyzeboard(board)
    if(x!=0):
        return (x*player)
    pos=-1
    value=-2
    for i in range(0,9):
        if(board[i]==0):
            board[i]=player
            score=-minimax(board,(player*-1))
            if(score>value):
                value=score
                pos=i
            board[i]=0

    if(pos==-1):
        return 0
    return value

def CompTurn(board):
    best_positions = []
    value=-2
    for i in range(0,9):
        if(board[i]==0):
            board[i]=1
            score=-minimax(board, -1)
            board[i]=0
            if(score>value):
                value=score
                best_positions=[i]
            elif(score==value):
                best_positions.append(i)

    pos = random.choice(best_positions)
    return pos + 1 # change to 1 indexing

def print_board(vector):
    """
    Convert a vector of 9 elements to a tic-tac-toe board display.
    
    Args:
        vector (list): List of 9 elements where -1 = "X", 1 = "O", 0 = empty
        
    Returns:
        str: String representation of the tic-tac-toe board
    """
    # Convert vector values to board symbols
    symbols = []
    for index, val in enumerate(vector):
        if val == -1:
            symbols.append("X")
        elif val == 1:
            symbols.append("O")
        elif val is None:
            symbols.append("?")
        else:
            symbols.append(str(index + 1))  # empty: show its position number
    
    # Create the board layout
    board = f"""
 {symbols[0]} | {symbols[1]} | {symbols[2]} 
-----------
 {symbols[3]} | {symbols[4]} | {symbols[5]} 
-----------
 {symbols[6]} | {symbols[7]} | {symbols[8]} 
"""
    
    print(board)

@parser.wrap()
def play(cfg: TicTacToeConfig) -> None:
    """Main game loop."""
    print("Loading policy checkpoint...")
    policy = make_policy(cfg.policy, ds_meta=cfg.metadata)
    print("Policy loaded.")

    i=0
    camera_failures = 0
    board_read_failures = 0
    while True:

        if i!=0:
            log_say("Now it is your turn", cfg.play_sounds)
            # Wait for Human to play
            busy_wait(cfg.player_turn_time_s)

        image = get_grid_image(camera_index = cfg.board_camera_index)
        if image is None:
            camera_failures += 1
            print(
                f"Board camera capture failed (index {cfg.board_camera_index}), attempt {camera_failures}/10. "
                "Close any other app using the camera, and run `python -m lerobot.find_cameras opencv` "
                "to check which index is the board camera, then pass --board_camera_index=<n>."
            )
            log_say("Camera capture failed", cfg.play_sounds)
            if camera_failures >= 10:
                print("Giving up after 10 failed board camera captures.")
                break
            busy_wait(1)
            continue
        camera_failures = 0
        # Send Gemini the whole camera view (640x480), uncropped and unwarped.
        # The old fixed crop/perspective points only matched one camera position.
        image.show()
        llm_output = get_LLM_output(image = image)

        # Keep what the board camera saw and what Gemini answered, so a wrong
        # reading can be diagnosed afterwards instead of guessed at.
        debug_dir = Path("board_debug")
        debug_dir.mkdir(exist_ok=True)
        image.save(debug_dir / f"board_turn_{i}.jpg")
        (debug_dir / f"board_turn_{i}.txt").write_text(llm_output, encoding="utf-8")
        print("--- Gemini's board reading ---")
        print(llm_output.strip())
        print("------------------------------")

        try:
            board_state = parse_board_state(llm_output)
        except BoardReadError as e:
            board_read_failures += 1
            print("Partial reading (? = could not tell; digits = empty cell, its position number):")
            print_board(e.partial)
            print(f"Board reading problem ({board_read_failures}/3): {e}")
            log_say("I could not read the board", cfg.play_sounds)
            if board_read_failures >= 3:
                print("Giving up after 3 unreadable board readings.")
                break
            continue
        board_read_failures = 0

        print(f"Board as Gemini reads it  (X = {X_COLOR}, the human; O = {O_COLOR}, the robot; digits = empty cell numbers):")
        print_board(board_state)

        if sum(board_state) > 1 or sum(board_state) < -1:
            print("Invalid Board State")
            log_say(f"Invalid Board State", cfg.play_sounds)
            break

        winner = analyzeboard(board_state)
        board_full = all(cell != 0 for cell in board_state)

        if winner != 0 or board_full:
            if winner == 1:
                print("Game Over: Robot (O) wins!")
                log_say("Game over, I win", cfg.play_sounds)
            elif winner == -1:
                print("Game Over: You (X) win!")
                log_say("Game over, you win", cfg.play_sounds)
            else:
                print("Game Over: Draw.")
                log_say("Game over, it's a draw", cfg.play_sounds)
            break

        comp_position = CompTurn(board_state)
        output = f"Place at Position {comp_position}"

        print(f"Decision: {output}")
        log_say(f"Placing at position {comp_position}", cfg.play_sounds)
        call_policy(cfg, instruction=output, policy=policy)
        
        i+=1

            

if __name__ == "__main__":
    play()