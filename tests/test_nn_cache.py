"""
Behaviour-neutrality tests for the per-search network evaluation cache.

The cache is only legitimate if it is invisible: the same search with and
without it must produce the same tree. These tests check that directly at a
small budget, which is far cheaper than the full golden master and catches a
broken cache key immediately.

The key is a digest of the encoded input tensor, which is the only thing the
network reads, so a hit provably returns the same policy and value the forward
pass would have. What these tests guard against is the cache being wired up
wrongly: a shared array mutated in place, a key that collapses distinct
positions, or state leaking between searches.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from golden import generate_golden as gg

from lucidtree.common.paths import get_project_root
from lucidtree.mcts.node import Node
from lucidtree.mcts.search import MCTS

MODEL_PATH = get_project_root() / "models" / f"{gg.MODEL}.pt"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"checkpoint {MODEL_PATH.name} not present (models/ is gitignored)",
)

NUM_SIMULATIONS = 384


@pytest.fixture(scope="module")
def searcher() -> MCTS:
    """A single-threaded CPU searcher shared across the tests."""
    torch.set_num_threads(1)
    return gg.make_searcher()


def _search(searcher: MCTS, name: str, *, use_nn_cache: bool) -> Node:
    """Search one golden position at a small budget."""
    moves = dict(gg.POSITIONS)[name]
    return gg.run_position(
        searcher,
        moves,
        num_simulations=NUM_SIMULATIONS,
        use_nn_cache=use_nn_cache,
    )


@pytest.mark.parametrize("name", ["opening_00", "opening_12", "ko", "capture"])
def test_cache_does_not_change_the_tree(searcher: MCTS, name: str) -> None:
    """Cached and uncached searches must agree exactly."""
    cached = _search(searcher, name, use_nn_cache=True)
    plain = _search(searcher, name, use_nn_cache=False)

    assert np.array_equal(cached.N, plain.N)
    assert np.allclose(cached.W, plain.W, rtol=0, atol=1e-9)
    assert np.allclose(cached.P, plain.P, rtol=0, atol=1e-9)


def test_cache_actually_hits_on_an_empty_board(searcher: MCTS) -> None:
    """A zero hit rate on an empty board would mean the key is wrong."""
    _search(searcher, "opening_00", use_nn_cache=True)
    stats = searcher.cache_stats

    assert stats["evaluations"] == NUM_SIMULATIONS + 1  # + the root
    assert stats["hits"] > 0
    assert stats["unique"] + stats["hits"] == stats["evaluations"]


def test_disabling_the_cache_reports_no_hits(searcher: MCTS) -> None:
    """With the cache off every evaluation is a real forward pass."""
    _search(searcher, "opening_00", use_nn_cache=False)
    stats = searcher.cache_stats

    assert stats["hits"] == 0
    assert stats["unique"] == stats["evaluations"]


def test_cache_does_not_persist_between_searches(searcher: MCTS) -> None:
    """Per-search scope: the second run must re-evaluate from scratch."""
    _search(searcher, "opening_00", use_nn_cache=True)
    first = dict(searcher.cache_stats)
    _search(searcher, "opening_00", use_nn_cache=True)
    second = dict(searcher.cache_stats)

    # A cache that survived would show far fewer unique positions the
    # second time round; identical counts mean it was rebuilt.
    assert first == second


def test_cached_policy_is_not_aliased(searcher: MCTS) -> None:
    """Nodes must not share the cached array; expand() mutates P in place."""
    root = _search(searcher, "opening_00", use_nn_cache=True)

    seen: list[np.ndarray] = [root.P]
    for child in root.children:
        if child is not None and child.is_expanded:
            seen.append(child.P)
    assert len(seen) > 1
    assert len({id(policy) for policy in seen}) == len(seen)
