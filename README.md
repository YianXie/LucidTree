# LucidTree

An implementation of Go AI, similar to [KataGo](https://github.com/lightvector/KataGo).

## Data/Model Download

For the processed and raw datasets, please check out this url on [Google Drive](https://drive.google.com/drive/folders/1Brh3DSuQ2fcs2gPFlytn4BvrR4j-qWHK?usp=sharing)

For the .pt model, please checkout this url on [Google Drive](https://drive.google.com/drive/folders/12OCXJz11Ely8U9kf6R822n4Apfg3QOef?usp=sharing)

## Sample Training Log

```log
2026-03-02 14:19:52 | INFO | training | Starting training
2026-03-02 14:19:52 | INFO | training | Using device: cuda
2026-03-02 14:19:52 | INFO | training | CUDA enabled: Tesla T4 (device cuda)
2026-03-02 14:19:52 | INFO | training | Total epoch = 10
2026-03-02 14:19:52 | INFO | training | Board size = 19
2026-03-02 14:19:52 | INFO | training | Batch size = 256
2026-03-02 14:19:52 | INFO | training | Gradient accumulation steps = 1
2026-03-02 14:19:53 | WARNING | training | Checkpoint file does not exist. Starting with no checkpoint file.
2026-03-02 14:21:55 | INFO | training | train_dataset length: 11910180
2026-03-02 14:21:55 | INFO | training | val_dataset length: 1484562
2026-03-02 14:21:55 | INFO | training | test_dataset length: 1490073
2026-03-02 14:21:55 | INFO | training | Finished loading train_loader.
2026-03-02 14:21:55 | INFO | training | Finished loading val_loader.
2026-03-02 14:21:55 | INFO | training | Finished loading test_loader.
2026-03-02 14:21:55 | INFO | training | Start training loop.
2026-03-02 14:21:55 | INFO | training | Epoch 0 started. Total batches: 46525 (grad accum steps: 1).
2026-03-02 14:24:41 | INFO | training | Epoch 0 | Batch 0 | loss = 6.4552 | total_loss = 6.4552
2026-03-03 00:54:26 | INFO | training | Epoch 0 | Batch 1000 | loss = 3.5255 | total_loss = 3998.0189
2026-03-03 11:26:00 | INFO | training | Epoch 0 | Batch 2000 | loss = 3.5116 | total_loss = 7600.1766
2026-03-03 21:53:35 | INFO | training | Epoch 0 | Batch 3000 | loss = 3.3946 | total_loss = 11102.3919
2026-03-04 08:22:14 | INFO | training | Epoch 0 | Batch 4000 | loss = 3.2518 | total_loss = 14536.7324
2026-03-04 18:52:55 | INFO | training | Epoch 0 | Batch 5000 | loss = 3.4532 | total_loss = 17913.6715
2026-03-05 05:24:18 | INFO | training | Epoch 0 | Batch 6000 | loss = 3.4546 | total_loss = 21251.8446
2026-03-05 15:57:44 | INFO | training | Epoch 0 | Batch 7000 | loss = 2.9904 | total_loss = 24553.4064
2026-03-06 02:26:40 | INFO | training | Epoch 0 | Batch 8000 | loss = 3.3198 | total_loss = 27835.5125
2026-03-06 12:56:47 | INFO | training | Epoch 0 | Batch 9000 | loss = 3.2537 | total_loss = 31088.1558
2026-03-06 23:28:27 | INFO | training | Epoch 0 | Batch 10000 | loss = 3.0867 | total_loss = 34325.1487
2026-03-07 09:57:54 | INFO | training | Epoch 0 | Batch 11000 | loss = 3.1091 | total_loss = 37540.0646
2026-03-07 20:27:09 | INFO | training | Epoch 0 | Batch 12000 | loss = 3.3522 | total_loss = 40732.7597
2026-03-08 06:56:17 | INFO | training | Epoch 0 | Batch 13000 | loss = 3.1251 | total_loss = 43916.3182
2026-03-08 17:26:02 | INFO | training | Epoch 0 | Batch 14000 | loss = 2.8750 | total_loss = 47082.0043
2026-03-09 03:55:07 | INFO | training | Epoch 0 | Batch 15000 | loss = 2.9189 | total_loss = 50228.6539
2026-03-09 14:24:48 | INFO | training | Epoch 0 | Batch 16000 | loss = 3.0630 | total_loss = 53356.6642
2026-03-10 00:55:10 | INFO | training | Epoch 0 | Batch 17000 | loss = 3.1561 | total_loss = 56473.7347
2026-03-10 11:26:21 | INFO | training | Epoch 0 | Batch 18000 | loss = 3.0193 | total_loss = 59570.3821
2026-03-10 21:55:32 | INFO | training | Epoch 0 | Batch 19000 | loss = 3.1720 | total_loss = 62658.3242
2026-03-11 08:25:20 | INFO | training | Epoch 0 | Batch 20000 | loss = 3.0200 | total_loss = 65737.8792
2026-03-11 18:53:50 | INFO | training | Epoch 0 | Batch 21000 | loss = 2.9343 | total_loss = 68793.9651
2026-03-12 05:23:29 | INFO | training | Epoch 0 | Batch 22000 | loss = 3.0230 | total_loss = 71840.4682
2026-03-12 15:54:07 | INFO | training | Epoch 0 | Batch 23000 | loss = 3.1741 | total_loss = 74879.6330
2026-03-13 02:22:23 | INFO | training | Epoch 0 | Batch 24000 | loss = 3.0876 | total_loss = 77904.2730
2026-03-13 12:57:30 | INFO | training | Epoch 0 | Batch 25000 | loss = 2.9856 | total_loss = 80919.5121
2026-03-13 23:28:49 | INFO | training | Epoch 0 | Batch 26000 | loss = 3.1521 | total_loss = 83922.1933
2026-03-14 09:58:00 | INFO | training | Epoch 0 | Batch 27000 | loss = 3.1709 | total_loss = 86916.3163
2026-03-14 20:26:12 | INFO | training | Epoch 0 | Batch 28000 | loss = 2.9535 | total_loss = 89904.3535
2026-03-15 06:55:18 | INFO | training | Epoch 0 | Batch 29000 | loss = 2.8643 | total_loss = 92879.9853
2026-03-15 17:24:51 | INFO | training | Epoch 0 | Batch 30000 | loss = 2.8073 | total_loss = 95836.0552
2026-03-16 03:53:55 | INFO | training | Epoch 0 | Batch 31000 | loss = 2.9543 | total_loss = 98789.0316
2026-03-16 14:22:48 | INFO | training | Epoch 0 | Batch 32000 | loss = 2.9952 | total_loss = 101733.0259
2026-03-17 00:52:50 | INFO | training | Epoch 0 | Batch 33000 | loss = 2.8964 | total_loss = 104664.6101
2026-03-17 11:21:26 | INFO | training | Epoch 0 | Batch 34000 | loss = 3.0877 | total_loss = 107591.2003
2026-03-17 21:49:56 | INFO | training | Epoch 0 | Batch 35000 | loss = 2.7083 | total_loss = 110512.6806
2026-03-18 08:17:47 | INFO | training | Epoch 0 | Batch 36000 | loss = 2.8176 | total_loss = 113419.9389
2026-03-18 18:45:55 | INFO | training | Epoch 0 | Batch 37000 | loss = 2.9288 | total_loss = 116326.2291
2026-03-19 05:15:05 | INFO | training | Epoch 0 | Batch 38000 | loss = 3.0942 | total_loss = 119220.8333
2026-03-19 15:44:53 | INFO | training | Epoch 0 | Batch 39000 | loss = 2.8944 | total_loss = 122110.3280
2026-03-20 02:15:51 | INFO | training | Epoch 0 | Batch 40000 | loss = 2.9244 | total_loss = 124985.5085
2026-03-20 12:47:23 | INFO | training | Epoch 0 | Batch 41000 | loss = 2.9609 | total_loss = 127858.7117
2026-03-20 23:18:03 | INFO | training | Epoch 0 | Batch 42000 | loss = 2.8284 | total_loss = 130725.0095
2026-03-21 09:45:58 | INFO | training | Epoch 0 | Batch 43000 | loss = 2.8915 | total_loss = 133585.1391
2026-03-21 20:14:24 | INFO | training | Epoch 0 | Batch 44000 | loss = 2.7208 | total_loss = 136435.7427
2026-03-22 06:42:28 | INFO | training | Epoch 0 | Batch 45000 | loss = 2.9194 | total_loss = 139281.2409
2026-03-22 17:11:05 | INFO | training | Epoch 0 | Batch 46000 | loss = 2.8271 | total_loss = 142117.2707
2026-03-25 12:17:49 | INFO | training | Found a better state at epoch 0
2026-03-25 12:17:49 | INFO | training | Epoch 0 finished | train_loss = 3.0866 | val_loss = 3.1268 | val_acc1 = 0.4325 | val_acc5 = 0.7619
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

### Week 8

Combined Monte Carlo Tree Search with Neural Network, similar to how AlphaZero works. It uses a PUCT (prior upper confidence score for trees) score instead of the ordinary UCT, (or "UCB"), in order to balance exploration and exploitation.

New features include:

- Combination of Monte Carlo Tree Search and Neural Network
- Stronger NN with more datasets
- Minor bug fixes for board.py

### Week 9 + 10 + 11

Improved Monte Carlo Tree Search to ensure that there is no logical errors. Moved training process to AWS EC2 for better efficiency and memory.

New features include:

- Optimized Monte Carlo Tree Search
- Better training settings for CUDA GPU

### API Implementation

Implemented a functional Django Rest Framework API in `/api` directory. It receives JSON input and send back a JSON output with the best move.

Sample JSON request:

```json
{
    "board_size": 9,
    "rules": "japanese",
    "komi": 6.5,
    "to_play": "B",
    "moves": [
        ["B", "D4"],
        ["W", "E4"]
    ],
    "algo": "mcts",
    "params": {
        "num_simulations": 300,
        "c_puct": 1.25
    }
}
```

Sample JSON response:

```json
{
    "best_move": "C5",
    "algorithm": "mcts",
    "stats": {
        "num_simulations": 300,
        "c_puct": 1.25,
        "time_ms": 91.83
    }
}
```

## Src File structure

```yaml
LucidTree/
├── src/
│   ├── lucidtree/
│   │   │── cli
│   │   │   ├── main.py                     # Python file for testing
│   │   │── common
│   │   │   ├── logging.py                  # Logger setup
│   │   │   ├── paths.py                    # Paths-related functions, such as getting project root
│   │   │── engine
│   │   │   ├── analysis.py                 # Function that handles a validated JSON input analysis request
│   │   │── go
│   │   │   ├── board.py                    # Python class that represents a Go game board
│   │   │   ├── game.py                     # Python class that represents a Go game, including board, players, and the winner
│   │   │   ├── move.py                     # Python class that represents a move in a game of Go
│   │   │   ├── player.py                   # Python class that represents a player in a game of Go
│   │   │   ├── rules.py                    # Python class that contains various rules for Go
│   │   │── mcts
│   │   │   ├── node.py                     # A custom Node data structure class used for Monte Carlo Tree Search
│   │   │   ├── search.py                   # A python program that searches for the most optimum move given the board and player
│   │   │── minimax
│   │   │   ├── search.py                   # A depth-limited MiniMax algorithm for Go with alpha-beta pruning
│   │   │── nn                              # All neural network related files
│   │   │   ├── datasets/
│   │   │   │   ├── gokifu_download.py      # A Python program that automatically downloads professional games from Gokifu website
│   │   │   │   ├── precomputed_dataset.py  # A class that represents a pre-computed dataset
│   │   │   │   ├── sgf_dataset.py          # A one-time running file that generates all the datasets
│   │   │   │   ├── sgf_parser.py           # An util file that parses SGF files and convert it to a Game object
│   │   │   ├── agent.py                    # The agent that loads the model and pick a move
│   │   │   ├── evaluate.py                 # A function that evaluate the training result based on the validation dataset
│   │   │   ├── features.py                 # Some features that are related to nn
│   │   │   ├── model.py                    # The PolicyValueNetwork CNN model
│   │   │   ├── split.py                    # Splits the game into training, validation, and testing set
│   │   │   ├── train.py                    # Runs the actual training with 30 epochs
│   │   │── constants.py                    # A file containing all the essential constants used in the project
```

## Django Rest Framework API

LucidTree uses Django Rest Framework (DRF) as its backend framework. All related files are located in the `/api` directory. To start the server, first follow the [development setup](#development) tutorial, then type the following command to start a local development server:

```bash
python manage.py makemigrations && python manage.py migrate
python manage.py runserver
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
git clone https://github.com/YianXie/LucidTree  # Clone this repository
cd LucidTree
uv init  # initialize the virtual environment
uv sync --dev  # install the dependencies
source .venv/bin/activate  # activate the virtual environment
```

Now you are ready to start developing. To see a quick demo, you may go the `main.py` and try a few different .sgf files or play your own.

> [!NOTE]
> Note: in some cases, the `main.py` file may not run correctly due to the `Module Not Found` error. In that case, try running the `make run` command at root level.

If there are any issues while developing, feel free to create an issue under the `issues` tab in the GitHub repository page.

Happy developing!

## Tests

This project contains some tests that you can run while developing to make sure everything works as expected.

To run tests:

```bash
uv init  # initialize the virtual environment if you haven't already done it
uv sync --dev  # install all the dependencies
```

```bash
make test  # Run at root level. This would run all tests.
# or
pytest  # Directly call the pytest command
```

To add more tests, simply add a new Python file in the `tests/` directory. Note that it must start with `test_xxx` or `xxx_test`

## CI/CD

This project uses GitHub Actions for continuous integration. The `ci.yml` workflow runs on every push and pull request, performing code quality checks including Ruff linting, Mypy type checking, isort import sorting validation, and pip-audit security scanning. The `tests.yml` workflow runs pytest tests on pushes to the main branch and all pull requests targeting main, ensuring that all tests pass before code is merged.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
