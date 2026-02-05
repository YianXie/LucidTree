# Mini-KataGo

A basic implementation of Go AI, similar to [KataGo](https://github.com/lightvector/KataGo).

## Data/Model Download

For the .pt datasets, please check out this url on [Google Drive](https://drive.google.com/drive/folders/1Brh3DSuQ2fcs2gPFlytn4BvrR4j-qWHK?usp=sharing)

For the .pt model, please checkout this url on [Google Drive](https://drive.google.com/drive/folders/12OCXJz11Ely8U9kf6R822n4Apfg3QOef?usp=sharing)

> [!NOTE]
> All .pt files ended with \_pol are policy-only, meaning they don't contain any value-network related data

## Training Result Overview

NN training graph:

![Training Graph](./assets/training_overview.png)

Sample training log:

```log
2026-02-04 21:27:00 | INFO | training | Starting training
2026-02-04 21:27:00 | INFO | training | Total epoch = 30
2026-02-04 21:27:00 | INFO | training | Board size = 9
2026-02-04 21:27:00 | INFO | training | Batch size = 256
2026-02-04 21:27:00 | INFO | training | train_dataset length: 484462
2026-02-04 21:27:00 | INFO | training | val_dataset length: 159855
2026-02-04 21:27:00 | INFO | training | test_dataset length: 161460
2026-02-04 21:27:01 | WARNING | training | Checkpoint file does not exist. Starting with no checkpoint file.
2026-02-04 21:29:05 | INFO | training | Epoch 0 finished | train_loss = 3.8185 | val_loss = 3.2441 | val_acc1 = 0.3797 | val_acc5 = 0.7475
2026-02-04 21:40:40 | INFO | training | Epoch 5 finished | train_loss = 2.7320 | val_loss = 3.0867 | val_acc1 = 0.4340 | val_acc5 = 0.8111
2026-02-04 21:52:35 | INFO | training | Epoch 10 finished | train_loss = 2.6108 | val_loss = 3.1063 | val_acc1 = 0.4437 | val_acc5 = 0.8184
2026-02-04 22:09:05 | INFO | training | Epoch 15 finished | train_loss = 2.5570 | val_loss = 3.1156 | val_acc1 = 0.4473 | val_acc5 = 0.8231
2026-02-04 23:28:42 | INFO | training | Epoch 20 finished | train_loss = 2.5254 | val_loss = 3.1386 | val_acc1 = 0.4505 | val_acc5 = 0.8237
2026-02-04 23:57:42 | INFO | training | Epoch 25 finished | train_loss = 2.5021 | val_loss = 3.1494 | val_acc1 = 0.4522 | val_acc5 = 0.8257
2026-02-05 00:11:29 | INFO | training | TEST | loss = 1.9065 | acc1 = 0.4478 | acc5 = 0.8210
2026-02-05 00:11:29 | INFO | training | Total training time: 5056.0261 seconds

```

## Timeline

### Week 1

Implemented basic Go board engine and a simple minimax file for tac-tac-toe that will be later used for Go as a depth-limited MiniMax (and probably see it fails badly)

New features include:

- Place move at specific position with specific color
- Captures detection
- Ko detection
- Score estimation at the end of the game
- Illegal move detection
- Display a real Go board with MatPlotLib

### Week 2

Implemented a basic depth-limited MiniMax algorithm for Go with alpha-beta pruning. It checks all possible moves in a given board state and choose the local optimal one by choosing the move that captures the most opponent's stones. Also did some minor updates to the board class.

New features include:

- Depth-limited Minimax algorithm with alpha-beta pruning
- Auto game-over when there are 2 consecutive passes
- Undo feature for game board

### Week 3 + 4

Implemented a basic Monte Carlo Tree Search for Go, as well as a Node class. The algorithm works by randomly choose legal position to play and calculate the UCT (Upper Confidence Bound applied to Trees). At the end, it picks the node with the most visits to ensure stability.

New features include:

- A basic Monte Carlo Tree Search for Go
- A new Node class data structure

### Week 5 + 6 + 7

Implemented a Convolution Neural Network (CNN) for Go with PyTorch. It works along with a policy network and a value network that allows the MCTS to perform better searching. Also refactored file structure so it's more sorted.

New features include:

- A decent Neural Network that learn from over 400 9\*9 Go .sgf files.
- Comprehensive logging when training
- Pre-computed dataset
- Model auto-saving

## Src File structure

```yaml
mini-katago/
├── src/                                    # All Python files
│   ├── mini_katago/                        # Go related files
│   │   │── go                              # All Python files related to Go, such as board.py and rules.py
│   │   │   ├── board.py                    # Python class that represents a Go game board
│   │   │   ├── game.py                     # Python class that represents a Go game, including board, players, and the winner
│   │   │   ├── move.py                     # Python class that represents a move in a game of Go
│   │   │   ├── player.py                   # Python class that represents a player in a game of Go
│   │   │   ├── rules.py                    # Python class that contains various rules for Go
│   │   │── mcts                            # All Python files related to Monte Carlo Tree Search, such as search.py
│   │   │   ├── node.py                     # A custom Node data structure class used for Monte Carlo Tree Search
│   │   │   ├── search.py                   # A python program that searches for the most optimum move given the board and player
│   │   │── misc                            # All Python files that are not absolutely essential to this project
│   │   │   ├── minimax.py                  # A depth-limited MiniMax algorithm for Go with alpha-beta pruning
│   │   │── nn                              # All neural network related files
│   │   │   ├── datasets/
│   │   │   │   ├── precomputed_dataset.py  # A class that represents a pre-computed dataset
│   │   │   │   ├── sgf_dataset.py          # A one-time running file that generates all the datasets
│   │   │   │   ├── sgf_parser.py           # An util file that parses SGF files and convert it to a Game object
│   │   │   ├── agent.py                    # The agent that loads the model and pick a move
│   │   │   ├── evaluate.py                 # A function that evaluate the training result based on the validation dataset
│   │   │   ├── model.py                    # The SmallPVNet CNN model
│   │   │   ├── play.py                     # A file that is mainly used for testing (e.g., nn v.s. nn and human v.s. nn)
│   │   │   ├── split.py                    # Splits the game into training, validation, and testing set
│   │   │   ├── train.py                    # Runs the actual training with 30 epochs
│   │   │── constants.py                    # A file containing all the essential constants used in the project
│   │   │── main.py                         # A file for testing
│   │   │── utils.py                        # Some utility functions
```

## Development

To start developing this project locally. Run the following command:

Install UV:

```bash
uv --version  # check if UV is already installed

# If it is not installed
curl -LsSf https://astral.sh/uv/install.sh | sh  # MacOS & Linux
# or
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
```

Clone the repository and setup:

```bash
git clone https://github.com/YianXie/Mini-KataGo  # Clone this repository
cd Mini-KataGo
uv init  # initialize the virtual environment
uv sync --dev  # install the dependencies
```

Now you are ready to start developing. To see a quick demo, you may go the `main.py` and try a few different .sgf files or play your own.

Happy developing!

## Tests

This project contains some tests that you can run while developing to make sure everything works as expected.

To run tests:

```bash
uv init  # initialize the virtual environment if you haven't already done it
uv sync --dev  # install all the dependencies
```

```bash
pytest  # Run at root level. This would run all tests.
```

To add more tests, simply add a new Python file in the `tests/` directory. Note that it must start with `test_xxx` or `xxx_test`

## CI/CD

This project uses GitHub Actions for continuous integration. The `ci.yml` workflow runs on every push and pull request, performing code quality checks including Ruff linting, Mypy type checking, isort import sorting validation, and pip-audit security scanning. The `tests.yml` workflow runs pytest tests on pushes to the main branch and all pull requests targeting main, ensuring that all tests pass before code is merged.
