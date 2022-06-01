# Dynamic Maze Solver (COMP6247)

This repository contains all the code used to solve the dynamic
maze challenge for COMP6247 Reinforcement & Online Learning.

## Usage

The configuration file (`config.py`) contains self-explanatory variables for configuring the program.

Run the following (from this directory) to execute the pretrained agent on a maze.
```bash
python -m src.eval  # run pretrained version
```

To train the agent, run the following (from this directory).
This will complete one epoch of training. 
```bash
python -m src.train  # run pretrained version
 ```

Alternatively, you can also run
```
make eval
```
or
```
make train
```
to execute the same functions.

Training make take a few minutes.

A copy of the q-table obtained for the optimal path has been
included in the `data/` directory.

---

## Repository Structure

The repository is structured with all the source code in the `src/` directory,
which is the further subdivided according to functionality.
The agent is implemented in the `agent/` directory.
Visualization functions can be found in `display/`.
Other additional functionality is in `lib/`.


## Log File

An agent can be logged to a file by calling the `log_agent()` function in the `lib` package.

The agent observation at every time-step is encoded in a text representation for easy logging. 

`X` marks the position of the agent. `W` denotes a wall.

A digit (`0-2`) denotes the life-time of a fire.

If the block is empty, it is an empty path.

Some examples are shown below.


```txt

agent surrounded by walls:

|W|W|W|

|W|X|W|

|W|W|W|


agent surrounded by empty paths with walls in the diagonals:

|W| |W|

| |X| |

|W| |W|


agent with one fire:

|W|W|W|

| |X|2|

|W|W|W|


agent with two fires:

|W|1|W|

|0|X|W|

|W| |W|

```