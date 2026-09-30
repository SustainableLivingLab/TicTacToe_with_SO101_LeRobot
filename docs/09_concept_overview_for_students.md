# 09. What This Project Is (Simple Explanation)

This page explains the TicTacToe robot project in simple English. No code
here. Just the ideas.

## The big idea

You already know pick-and-place. In your bootcamp, the robot arm learned to
pick up a pen and put it down in a new spot.

This project is the same idea. The robot arm picks up a token and puts it
down in a new spot. But we add one small extra piece: **the robot must
choose the correct spot itself.**

That is the whole project in one sentence:

> Pick-and-place, plus a simple decision about where to place.

## The game: Tic-Tac-Toe

Tic-Tac-Toe is a game with a 3x3 grid. Two players take turns. One player
uses X. The other player uses O. The first player to get three in a row
(across, down, or diagonal) wins.

In this project:

- The **human** always plays **X**. The human always moves first.
- The **robot** always plays **O**. The robot always moves second.

This is fixed. It does not change during the game.

## Three jobs, three different tools

Making the robot play Tic-Tac-Toe needs three separate jobs. Each job uses
a different tool. This is important to understand: **the robot arm itself
does not "think" about the game.** Thinking and moving are two separate
things.

### Job 1: See the board (a camera + Gemini)

A camera takes a picture of the board. We send that picture to Gemini (a
vision AI model). We ask Gemini one simple question: "What is in each of
the 9 cells? Empty, X, or O?"

Gemini's only job is to describe the picture in words. Gemini does not
choose a move. Gemini does not know whose turn it is. Gemini just reports
what it sees.

### Job 2: Decide the move (minimax, a simple algorithm)

Once we know the board (empty/X/O in each cell), we need to choose the
robot's next move.

This part uses **no AI at all.** It uses a classic algorithm called
**minimax**. Minimax checks every possible next move, and every possible
reply after that, all the way to the end of the game. It picks the move
that gives the robot the best result.

Tic-Tac-Toe is small enough that a computer can check every possibility
almost instantly. This is why we do not need AI here. A simple, exact
algorithm is enough.

### Job 3: Move the arm (the trained robot policy, ACT)

Now the robot knows *where* to place the token (for example, "cell 5").
This job is the pick-and-place part, exactly like the pen exercise. The
robot arm needs to:

1. Reach down and pick up a token.
2. Move to the correct cell.
3. Place the token down.

This part **is** trained with AI, using a policy called **ACT** (Action
Chunking Transformer). This is the part that connects most closely to what
you already learned with the pen.

## How is this different from the pen exercise?

In the pen exercise, the robot only ever learned **one** pick-and-place
motion: pick up the pen, put it in one place.

In this project, the robot needs **nine** different pick-and-place
motions: one for each of the 9 grid cells. The robot needs to know which
one of the nine to do, every time it moves.

This is the "small extra piece" from the big idea above. We need a way to
tell the trained robot: "This time, use motion number 5, not motion number
3."

## How the robot knows which of the 9 motions to use

This is the most important idea in the whole project. Read this part
slowly.

When we record training data, we do not just record "pick up token, put it
down." We also record a short **instruction**, like a label, together with
every single recording. The instruction is a simple sentence:

```
"Place at Position 5"
```

We do this 9 times, once for each cell, so we get 9 different instructions:
`"Place at Position 1"`, `"Place at Position 2"`, all the way to
`"Place at Position 9"`.

During training, the AI model (ACT) learns to connect **two things
together**:

- the instruction ("Place at Position 5")
- the correct arm motion for that instruction

This is called **task conditioning**. "Task" means the instruction. "Conditioning"
means the model's behavior changes depending on the instruction it
receives. Same robot, same trained model, but it moves differently
depending on which instruction you give it.

So at game time, the full sentence is:

1. Camera + Gemini: "Cell 5 is empty, other cells have X and O in them."
2. Minimax: "Best move for the robot is cell 5."
3. The instruction `"Place at Position 5"` is sent to the trained ACT
   model.
4. ACT, having learned all 9 motions during training, performs the
   correct one: the pick-and-place for cell 5.

## A simple way to say it to someone else

If someone asks you "how does this robot know where to put the piece,"
you can answer like this:

> "A camera and a vision AI read the board. A simple algorithm called
> minimax decides the best move, the same way a chess computer thinks
> ahead. That move becomes a short instruction, like 'place at position
> 5.' A separately trained robot model then performs the correct
> pick-and-place motion for that instruction, because it learned all 9
> possible motions during training, one per grid cell."

## One thing that makes this project harder than the pen exercise

In the pen exercise, there was only one motion to learn, so training data
only needed to teach one thing.

Here, we need the robot to learn 9 different motions and correctly
match each one to its instruction. This needs more recorded data (this
project records 10 examples for each of the 9 cells, 90 recordings total),
and the model must be built to accept an instruction as input, not only
camera images and joint positions.

## Where to look next

- `00_overview.md`: the same ideas, written for a technical/engineering
  reader.
- `03_dataset_and_training.md`: exactly how the 90 recordings are planned
  and collected.
- `06_act_configuration.md`: the technical detail of how the instruction
  ("Place at Position N") is turned into something the ACT model can
  actually use internally.
- `04_gameplay_pipeline.md`: the full step-by-step of one robot turn,
  written for someone reading the code.
