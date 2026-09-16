# 01. Hardware Setup

## Bill of materials and 3D printing

Source parts and print instructions: [TheRobotStudio/SO-ARM100
README](https://github.com/TheRobotStudio/SO-ARM100). This covers the full
bill of materials and links to source each part, plus 3D printing guidance.

You need two arms: one leader (teleoperation input), one follower (executes
actions). Both share the same printed parts and motor count, only the gear
ratios differ on some joints.

## Motors

Both arms use 6x STS3215 servos, one per joint.

| Follower-arm axis | Motor ID | Gear ratio |
|---|---|---|
| Base / shoulder pan | 1 | 1/345 |
| Shoulder lift | 2 | 1/345 |
| Elbow flex | 3 | 1/345 |
| Wrist flex | 4 | 1/345 |
| Wrist roll | 5 | 1/345 |
| Gripper | 6 | 1/345 |

| Leader-arm axis | Motor ID | Gear ratio |
|---|---|---|
| Base / shoulder pan | 1 | 1/191 |
| Shoulder lift | 2 | 1/345 |
| Elbow flex | 3 | 1/191 |
| Wrist flex | 4 | 1/147 |
| Wrist roll | 5 | 1/147 |
| Gripper | 6 | 1/147 |

The leader uses lower gear ratios on most joints so it can be moved by hand
without much resistance. The follower uses the higher 1/345 ratio throughout
for the torque needed to carry real loads (in this project: picking up a
wooden tile piece).

## Assembly

Full step-by-step assembly (joint by joint, with video for each step) is in
`TicTacToe_with_SO101/docs/source/so101.mdx`. Summary of the sequence:

1. Clean 3D-printed parts (remove support material).
2. Assemble joints 1 through 5 in order, each joint mounting its motor,
   motor horns, and the next structural part.
3. Assemble the gripper (follower) or handle (leader). These differ: the
   follower gets a gripper claw, the leader gets a handle with a trigger
   that operates the follower's gripper during teleoperation.

## Software prerequisite for hardware

Install LeRobot with the Feetech SDK extra (required for STS3215 motor
communication):

```bash
pip install -e ".[feetech]"
```

## Motor configuration

Each motor needs a unique ID and a shared baudrate set once, written to the
motor's EEPROM. This has to be done per motor, individually, before final
assembly of the motor chain.

1. Find the USB port for each arm's controller board:

```bash
python -m lerobot.find_port
```

Follow the prompt: disconnect the arm being identified when asked, and the
script reports which port corresponds to it.

2. Set motor IDs and baudrate, one motor at a time, starting with the
   gripper:

```bash
# Follower
python -m lerobot.setup_motors \
    --robot.type=so101_follower \
    --robot.port=<port from step 1>

# Leader
python -m lerobot.setup_motors \
    --teleop.type=so101_leader \
    --teleop.port=<port from step 1>
```

The script prompts you to connect each motor individually (starting with
`gripper`, then `wrist_roll`, and so on) to the controller board, with only
that one motor connected at a time. Press Enter after each connection; the
script assigns the correct ID and baudrate automatically. Once done, chain
all motors together via their 3-pin cables and connect the first motor
(shoulder pan) to the controller board.

## Calibration

Calibration aligns the leader and follower arms so the same joint-position
values mean the same physical position on both arms. This matters because a
policy trained on the follower's joint states needs those values to be
meaningful and repeatable.

```bash
# Follower
python -m lerobot.calibrate \
    --robot.type=so101_follower \
    --robot.port=<port> \
    --robot.id=<name for this arm>

# Leader
python -m lerobot.calibrate \
    --teleop.type=so101_leader \
    --teleop.port=<port> \
    --teleop.id=<name for this arm>
```

The calibration routine asks you to move the arm to its middle position
first, then move each joint through its full range of motion.

## Camera rig

This project uses a single fixed camera (`camera_index = 2` in
`play_TicTacToe.py`) positioned to view the board at an angle, not directly
overhead. The game loop corrects for this with a perspective transform:

- Raw camera frame is cropped to the board region using fixed percentage
  offsets.
- A perspective warp (`cv2.getPerspectiveTransform` / `cv2.warpPerspective`)
  maps 4 manually measured corner points to a clean top-down 400x400 view.
- A final crop and 180-degree rotation produce the frame sent to Gemini.

All of these values (`camera_index`, crop percentages, the 4 corner points)
are hardcoded and calibrated to one specific physical camera position and
board placement. If the camera, table, or board moves, these values need to
be re-measured. `TicTacToe_with_SO101/src/lerobot/scripts/ticTacToe/image_transformation_testing.py`
is a standalone script for testing and recalibrating this pipeline without
running the full game loop.

## Physical game pieces

This project's board (wooden tiles slotted into a fixed 3x3 frame, each
pre-painted with either an X or an O) ships 5 X pieces and 4 O pieces (9
total). Blue = X (human), Red = O (robot). This matches the maximum
occupancy of a real game: X always moves first, so a full or drawn board
always has exactly one more X than O (5X + 4O = 9 cells).

Maximum board occupancy used anywhere in this project's dataset or gameplay
logic is 4 X + 4 O (one move before a full board), so the 5th X piece is
never used during data collection. See `03_dataset_and_training.md` for the
occupancy scenarios used during data collection.
