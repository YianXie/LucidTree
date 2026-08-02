"""
Equivalence test for the vectorized PUCT selection in ``lucidtree.mcts.node``.

``_select_action_scalar`` below is the pre-optimization implementation, frozen
here verbatim (including its numeric promotions) so it keeps working as an
oracle even if ``Node.Q``/``Node.U`` are later touched. Every randomized state
must produce the same action index from both implementations, and the
per-action Q and U terms must agree bit for bit.

The randomized states deliberately over-sample the cases where a vectorized
rewrite is most likely to diverge:

* all-zero visit counts (every Q is the unvisited 0.0, U alone decides)
* exact score ties from quantized priors (tie-break must pick the lowest index)
* single-legal and zero-legal masks
* negative W (the Q term goes below the unvisited default)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lucidtree.constants import BOARD_SIZE, INFINITY
from lucidtree.go.board import Board
from lucidtree.go.player import Player
from lucidtree.mcts.node import Node

TOTAL_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1
NUM_STATES = 10_000


def _q_scalar(node: Node, action: int) -> float:
    """Original ``Node.Q``."""
    total_visit_count = node.N[action]
    if total_visit_count == 0:
        return 0.0
    total_value_sum = node.W[action]
    return float(node.value_weight * (total_value_sum / total_visit_count))


def _u_scalar(node: Node, action: int, c_puct: float) -> float:
    """Original ``Node.U``."""
    sum_visits = node.N.sum()
    prior = node.P[action]
    action_visits = node.N[action]
    return float(
        node.policy_weight
        * c_puct
        * prior
        * (math.sqrt(sum_visits) / (1.0 + action_visits))
    )


def _select_action_scalar(node: Node, c_puct: float) -> np.int64:
    """Original ``Node.select_action``."""
    best_score = -INFINITY
    best_action: np.int64 = np.int64(BOARD_SIZE * BOARD_SIZE)
    legal_actions = np.where(node.legal_mask)[0]
    for action in legal_actions:
        score = _q_scalar(node, action) + _u_scalar(node, action, c_puct)
        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def _make_node() -> Node:
    """A node whose arrays we overwrite; the board itself is never read."""
    black = Player.black()
    white = Player.white()
    black.opponent = white
    white.opponent = black
    return Node(board=Board(BOARD_SIZE, black, white), to_play=black)


def _random_state(node: Node, rng: np.random.Generator, index: int) -> float:
    """
    Overwrite one node's statistics with a randomized state.

    Args:
        node (Node): the node to overwrite in place.
        rng (np.random.Generator): the seeded generator.
        index (int): the trial number, used to pick the regime.

    Returns:
        float: the c_puct to search this state with.
    """
    regime = index % 8
    legal = rng.random(TOTAL_ACTIONS) < rng.choice([0.02, 0.3, 0.9, 1.0])

    if regime == 0:
        # Fresh node: nothing visited, so every Q is the unvisited default.
        n = np.zeros(TOTAL_ACTIONS, dtype=np.int32)
        w = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
        p = rng.random(TOTAL_ACTIONS).astype(np.float32)
    elif regime == 1:
        # Total tie: identical priors, no visits. Lowest index must win.
        n = np.zeros(TOTAL_ACTIONS, dtype=np.int32)
        w = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
        p = np.full(TOTAL_ACTIONS, 1.0 / TOTAL_ACTIONS, dtype=np.float32)
    elif regime == 2:
        # Heavily quantized: many exact ties in both the Q and U terms.
        n = rng.integers(0, 3, TOTAL_ACTIONS).astype(np.int32)
        w = rng.choice([-1.0, 0.0, 1.0], TOTAL_ACTIONS).astype(np.float32)
        p = rng.choice([0.0, 0.25, 0.5], TOTAL_ACTIONS).astype(np.float32)
    elif regime == 3:
        # Exactly one legal action.
        legal = np.zeros(TOTAL_ACTIONS, dtype=np.bool_)
        legal[int(rng.integers(0, TOTAL_ACTIONS))] = True
        n = rng.integers(0, 50, TOTAL_ACTIONS).astype(np.int32)
        w = rng.normal(0.0, 1.0, TOTAL_ACTIONS).astype(np.float32)
        p = rng.random(TOTAL_ACTIONS).astype(np.float32)
    elif regime == 4:
        # No legal actions at all (board full).
        legal = np.zeros(TOTAL_ACTIONS, dtype=np.bool_)
        n = rng.integers(0, 50, TOTAL_ACTIONS).astype(np.int32)
        w = rng.normal(0.0, 1.0, TOTAL_ACTIONS).astype(np.float32)
        p = rng.random(TOTAL_ACTIONS).astype(np.float32)
    elif regime == 5:
        # Deep node: large visit counts, strongly negative values.
        n = rng.integers(0, 20_000, TOTAL_ACTIONS).astype(np.int32)
        w = -rng.random(TOTAL_ACTIONS).astype(np.float32) * n
        p = rng.dirichlet(np.full(TOTAL_ACTIONS, 0.3)).astype(np.float32)
    elif regime == 6:
        # Mixed: some actions visited, most not.
        n = (rng.random(TOTAL_ACTIONS) < 0.1).astype(np.int32) * rng.integers(
            1, 100, TOTAL_ACTIONS
        ).astype(np.int32)
        w = rng.normal(0.0, 2.0, TOTAL_ACTIONS).astype(np.float32) * n
        p = rng.dirichlet(np.full(TOTAL_ACTIONS, 0.03)).astype(np.float32)
    else:
        # Zero priors everywhere: U vanishes and Q alone decides.
        n = rng.integers(0, 8, TOTAL_ACTIONS).astype(np.int32)
        w = rng.choice([-2.0, 0.0, 2.0], TOTAL_ACTIONS).astype(np.float32)
        p = np.zeros(TOTAL_ACTIONS, dtype=np.float32)

    node.legal_mask = legal
    node.N = n
    node.W = w.astype(np.float32)
    node.P = p
    node.policy_weight = float(rng.choice([1.0, 0.5, 2.0]))
    node.value_weight = float(rng.choice([1.0, 0.25, 3.0]))
    return float(rng.choice([1.25, 1.5, 0.0, 4.0]))


def test_select_action_matches_scalar_implementation() -> None:
    """The vectorized selection must pick the same action every single time."""
    node = _make_node()
    rng = np.random.default_rng(20260801)

    mismatches = []
    for i in range(NUM_STATES):
        c_puct = _random_state(node, rng, i)
        want = _select_action_scalar(node, c_puct)
        got = node.select_action(c_puct=c_puct)
        if int(got) != int(want):
            mismatches.append((i, int(want), int(got)))
            if len(mismatches) > 5:
                break

    assert not mismatches, (
        f"{len(mismatches)} state(s) selected a different action; "
        f"first few (trial, scalar, vectorized): {mismatches[:5]}"
    )


def test_q_and_u_terms_are_bit_identical() -> None:
    """The vectorized Q and U terms must be bit-identical, not merely close."""
    node = _make_node()
    rng = np.random.default_rng(778899)

    for i in range(500):
        c_puct = _random_state(node, rng, i)
        q_vec = node.Q_all()
        u_vec = node.U_all(c_puct)
        for action in np.where(node.legal_mask)[0]:
            assert q_vec[action] == _q_scalar(node, int(action))
            assert u_vec[action] == _u_scalar(node, int(action), c_puct)


def test_scalar_q_still_agrees_with_vector_q() -> None:
    """``Node.Q`` (public API) and ``Node.Q_all`` must not drift apart."""
    node = _make_node()
    rng = np.random.default_rng(31337)

    for i in range(200):
        _random_state(node, rng, i)
        q_vec = node.Q_all()
        for action in range(TOTAL_ACTIONS):
            assert q_vec[action] == node.Q(action)


@pytest.mark.parametrize("c_puct", [0.0, 1.25, 1.5, 4.0])
def test_ties_break_to_lowest_legal_index(c_puct: float) -> None:
    """With every legal action tied, the lowest legal index must win."""
    node = _make_node()
    node.legal_mask = np.zeros(TOTAL_ACTIONS, dtype=np.bool_)
    node.legal_mask[[7, 42, 99, 200, 361]] = True
    node.N = np.zeros(TOTAL_ACTIONS, dtype=np.int32)
    node.W = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
    node.P = np.full(TOTAL_ACTIONS, 0.125, dtype=np.float32)

    assert int(node.select_action(c_puct=c_puct)) == 7
    assert int(_select_action_scalar(node, c_puct)) == 7


def test_no_legal_actions_returns_pass_index() -> None:
    """A node with no legal actions keeps the scalar fallback of index 361."""
    node = _make_node()
    node.legal_mask = np.zeros(TOTAL_ACTIONS, dtype=np.bool_)
    node.N = np.arange(TOTAL_ACTIONS, dtype=np.int32)
    node.W = np.ones(TOTAL_ACTIONS, dtype=np.float32)
    node.P = np.ones(TOTAL_ACTIONS, dtype=np.float32)

    assert int(node.select_action(c_puct=1.25)) == BOARD_SIZE * BOARD_SIZE
    assert int(_select_action_scalar(node, 1.25)) == BOARD_SIZE * BOARD_SIZE
