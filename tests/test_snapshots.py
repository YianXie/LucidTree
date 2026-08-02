"""
Tests for the power-of-two snapshot hook.

The point of the hook is that one run to N_max yields the visit counts for
every smaller budget, so these tests check both halves of that claim:

1. Enabling snapshots does not change the search. The final ``root.N`` from a
   snapshotted run must equal the golden master exactly, since a snapshot is
   only a copy taken between simulations.
2. The snapshot taken at budget k really is the tree after k simulations -
   verified against a separate search actually run with num_simulations=k.
"""

from __future__ import annotations

# fmt: off
import numpy as np
import pytest
import torch
from golden import generate_golden as gg

from lucidtree.common.paths import get_project_root
from lucidtree.mcts.search import MCTS

# fmt: on

MODEL_PATH = get_project_root() / "models" / f"{gg.MODEL}.pt"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"checkpoint {MODEL_PATH.name} not present (models/ is gitignored)",
)

BUDGET = 512


@pytest.fixture(scope="module")
def searcher() -> MCTS:
    """A single-threaded CPU searcher shared across the tests."""
    torch.set_num_threads(1)
    return gg.make_searcher()


@pytest.mark.parametrize("name", ["opening_00", "ko"])
def test_snapshots_do_not_change_the_search(searcher: MCTS, name: str) -> None:
    """The final tree must be identical whether or not snapshots are on."""
    moves = dict(gg.POSITIONS)[name]
    with_snaps = gg.run_position(
        searcher, moves, num_simulations=BUDGET, snapshot_powers_of_two=True
    )
    without = gg.run_position(searcher, moves, num_simulations=BUDGET)

    assert np.array_equal(with_snaps.N, without.N)
    assert np.allclose(with_snaps.W, without.W, rtol=0, atol=1e-9)
    assert np.allclose(with_snaps.P, without.P, rtol=0, atol=1e-9)


def test_snapshot_budgets_are_powers_of_two(searcher: MCTS) -> None:
    """Snapshots land on 1, 2, 4, ... and each holds exactly that many visits."""
    moves = dict(gg.POSITIONS)["opening_00"]
    root = gg.run_position(
        searcher, moves, num_simulations=BUDGET, snapshot_powers_of_two=True
    )

    assert root.snapshots is not None
    budgets = sorted(root.snapshots)
    assert budgets == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for budget in budgets:
        assert int(root.snapshots[budget].sum()) == budget


def test_snapshot_matches_a_real_run_at_that_budget(searcher: MCTS) -> None:
    """The snapshot at k must equal a search actually run with N=k."""
    moves = dict(gg.POSITIONS)["opening_00"]
    root = gg.run_position(
        searcher, moves, num_simulations=BUDGET, snapshot_powers_of_two=True
    )
    assert root.snapshots is not None

    for budget in (1, 8, 64, 256):
        short = gg.run_position(searcher, moves, num_simulations=budget)
        assert np.array_equal(root.snapshots[budget], short.N), (
            f"snapshot at N={budget} differs from a real N={budget} run"
        )


def test_snapshots_are_copies(searcher: MCTS) -> None:
    """A snapshot must not be a live view of root.N."""
    moves = dict(gg.POSITIONS)["opening_00"]
    root = gg.run_position(
        searcher, moves, num_simulations=BUDGET, snapshot_powers_of_two=True
    )
    assert root.snapshots is not None

    assert int(root.snapshots[1].sum()) == 1
    assert int(root.N.sum()) == BUDGET
    assert all(snap is not root.N for snap in root.snapshots.values())


def test_snapshots_default_to_off(searcher: MCTS) -> None:
    """The default path is untouched, so root.snapshots stays None."""
    moves = dict(gg.POSITIONS)["opening_00"]
    root = gg.run_position(searcher, moves, num_simulations=64)

    assert root.snapshots is None
