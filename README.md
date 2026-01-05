# Mini-KataGo

A basic implementation of Go AI, similar to [KataGo](https://github.com/lightvector/KataGo).

## Timeline

### Week 1

Implemented basic Go board engine and a simple minimax file for tac-tac-toe that will be later used for Go as a depth-limited MiniMax (and probably see it fails badly)

New features include:

-   Place move at specific position with specific color
-   Captures detection
-   Ko detection
-   Score estimation at the end of the game
-   Illegal move detection
-   Display a real Go board with MatPlotLib

### Week 2

Implemented a basic depth-limited MiniMax algorithm for Go with alpha-beta pruning. It checks all possible moves in a given board state and choose the local optimal one by choosing the move that captures the most opponent's stones. Also did some minor updates to the board class.

New features include:

-   Depth-limited Minimax algorithm with alpha-beta pruning
-   Auto game-over when there are 2 consecutive passes
-   Undo feature for game board

### Week 3 + 4

Implemented a basic Monte Carlo Tree Search for Go, as well as a Node class. The algorithm works by randomly choose legal position to play and calculate the UCT (Upper Confidence Bound applied to Trees). At the end, it picks the node with the most visits to ensure stability.

New features include:

-   A basic Monte Carlo Tree Search for Go
-   A new Node class data structure

## File structure

```
mini-katago/
├── .github/
│   ├── workflows
│       ├── ci.yml
│       ├── tests.yml
├── src/                        # All Python files
│   ├── mini_katago/            # Go related files
│   │   │── __init__.py
│   │   │── board.py            # A file that represents the Go game board
│   │   │── constants.py        # A file containing all the essential constants used in the project
│   │   │── main.py             # A file for testing
│   │   │── mcts.py             # A Monte Carlo Go Simulation file (also refer to as Monte Carlo Tree Search)
│   │   │── minimax.py          # A simple depth-limited minimax algorithm for Go
│   │   │── player.py           # A file representing the player
│   │   │── rules.py            # A file that contains some basic rules for Go
├── tests/                      # All unit-tests
│   ├── test_board_rules.py     # Test if board rules works correctly
├── .gitignore
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
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
