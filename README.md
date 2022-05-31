# Dynamic Maze Solver (COMP6247)

This repository contains all the code used to solve the dynamic
maze challenge for COMP6247 Reinforcement & Online Learning.

[//]: # (todo - create config file)
[//]: # (todo - write up training)
[//]: # (todo - write up inference/running)
[//]: # (todo - write up directory structure)
[//]: # (todo - requirements.txt)

## Usage

Run the following to execute a pretrained agent on a maze.
```bash
python eval.py
```

To train the agent, run the following.
This will complete one epoch of training. 
```bash
python train.py # run training
```


Training takes approximately _ hours on.

A copy of the q-table obtained for the optimal path has been
included in the `results/` directory.

---

## Repository Structure

```bash

```

## Pretrained Agents

The pretrained Q-table for an agent can be loaded in directly.

A Q-table for the optimal agent is provided in the `pretrained-agents/` directory.

Running the following will run this agent on the maze.

```bash
python pretrained.py
```

## Log File

An agent can be logged to a file by calling the `log_agent()` function in the `lib` package.

[//]: # (The agent observation at every time-step is encoded in a text representation for easy logging. )

[//]: # (`X` marks the position of the agent. `W` denotes a wall.)

[//]: # (A digit &#40;`0-2`&#41; denotes the life-time of a fire.)

[//]: # (If the block is empty, it is an empty path.)

[//]: # (Some examples are shown below.)

[//]: # ()
[//]: # (```txt)

[//]: # (agent surrounded by walls:)

[//]: # (|W|W|W|)

[//]: # (|W|X|W|)

[//]: # (|W|W|W|)

[//]: # ()
[//]: # (agent surrounded by empty paths with walls in the diagonals:)

[//]: # (|W| |W|)

[//]: # (| |X| |)

[//]: # (|W| |W|)

[//]: # ()
[//]: # (agent with one fire:)

[//]: # (|W|W|W|)

[//]: # (| |X|2|)

[//]: # (|W|W|W|)

[//]: # ()
[//]: # (agent with two fires:)

[//]: # (|W|1|W|)

[//]: # (|0|X|W|)

[//]: # (|W| |W|)

[//]: # (```)