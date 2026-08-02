"""
Measure LucidTree MCTS throughput. Run from the repo root:

    uv run python bench_throughput.py
    uv run python bench_throughput.py 256 1024      # override the N sweep

Everything in the study's compute plan scales off the number this prints.
Run it on the machine you'll actually collect data on, plugged in, not on battery
(macOS throttles aggressively on battery).

Device policy: CPU, single-threaded, always. MPS measured slower at every budget
for this network (batch-1 kernel launch overhead exceeds compute), and thread
count changes convolution reduction order, which can perturb float results. The
golden master is generated single-threaded, so every measurement here must match
that configuration or the numbers describe a different program.
"""

import sys
import time

import torch

from lucidtree.constants import BOARD_SIZE, KOMI, RULES
from lucidtree.go.board import Board
from lucidtree.go.player import Player
from lucidtree.mcts.search import MCTS

# ---- Settings to match your study's control variables -----------------------
C_PUCT = 1.25  # NOTE: analysis.py defaults to 1.5 -- set this explicitly
DIRICHLET_ALPHA = 0.0
DIRICHLET_EPSILON = 0.0
MODEL = "latest"
# -----------------------------------------------------------------------------


def make_empty_position() -> tuple[Board, Player]:
    """Empty 19x19 board, Black to play. Worst case for legal-move count."""
    black = Player.black()
    white = Player.white()
    black.opponent = white
    white.opponent = black
    board = Board(BOARD_SIZE, black, white)
    return board, black


def bench(mcts: MCTS, n_sims: int, label: str) -> float:
    board, to_play = make_empty_position()
    t0 = time.perf_counter()
    mcts.run(
        board=board,
        to_play=to_play,
        num_simulations=n_sims,
        c_puct=C_PUCT,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=DIRICHLET_EPSILON,
        komi=KOMI,
        rules=RULES,
        # max_time_ms deliberately NOT passed -- if it is set > 0 it silently
        # replaces N as the independent variable.
    )
    dt = time.perf_counter() - t0
    rate = n_sims / dt
    print(f"  {label:>12}  N={n_sims:>6}  {dt:7.2f}s  {rate:8.1f} sims/sec")
    return rate


def main() -> None:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    print(f"Device: {device} (threads={torch.get_num_threads()})\n")

    sweep = [int(a) for a in sys.argv[1:]] or [256, 1024, 4096, 16384]

    mcts = MCTS(model=MODEL, device=device)

    # Warm up: first call pays model load + kernel compilation costs.
    print("Warming up...")
    bench(mcts, 128, "warmup")
    print()

    print("Measurements (throughput usually DROPS as N grows -- deeper trees")
    print("mean more Python work per simulation):")
    rates = {}
    for n in sweep:
        rates[n] = bench(mcts, n, "empty board")

    r = rates[sweep[-1]]
    print(f"\nUse S = {r:.0f} sims/sec for planning.\n")

    # --- Study cost projections -------------------------------------------
    positions = 200
    n_max = 65536
    hours = positions * n_max / r / 3600
    print(
        f"SQ1-SQ4 ({positions} positions, one run each to N={n_max}): "
        f"{hours:.1f} CPU-hours"
    )

    print("\nSQ5 game costs (per game, ~250 moves, 2N vs N):")
    for n_low in (32, 128, 512, 2048):
        sims_per_game = 375 * n_low  # (250/2) * (N + 2N)
        secs = sims_per_game / r
        print(
            f"  {n_low:>5} vs {2 * n_low:>5}: {secs / 60:6.1f} min/game  "
            f"-> {600 * secs / 3600:6.1f} hours for 600 games"
        )


if __name__ == "__main__":
    main()
