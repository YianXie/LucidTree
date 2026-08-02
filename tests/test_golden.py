"""
Golden-master test for the MCTS search.

Regenerates the root statistics for every fixed position in
``tests/golden/generate_golden.py`` and compares them against the committed
``tests/golden/golden.npz``.

``root.N`` must be **exactly** equal: the visit counts are integers, so any
difference at all means the search visited a different node and is therefore a
different algorithm, not a faster implementation of the same one. ``root.W``
and ``root.P`` are compared with ``rtol=0, atol=1e-9``, which for these
magnitudes is bit-identity in all but name.

This test is slow by construction (20 positions x 2048 simulations). It is
skipped when ``models/latest.pt`` is absent, e.g. in CI, since the checkpoint
is gitignored.
"""

from __future__ import annotations

import numpy as np
import pytest
from golden import generate_golden as gg

from lucidtree.common.paths import get_project_root

MODEL_PATH = get_project_root() / "models" / f"{gg.MODEL}.pt"

pytestmark = [
    pytest.mark.skipif(
        not MODEL_PATH.exists(),
        reason=f"checkpoint {MODEL_PATH.name} not present (models/ is gitignored)",
    ),
    pytest.mark.skipif(
        not gg.GOLDEN_PATH.exists(),
        reason="golden.npz missing; run tests/golden/generate_golden.py",
    ),
]

POSITION_NAMES = [name for name, _ in gg.POSITIONS]


@pytest.fixture(scope="module")
def expected() -> dict[str, np.ndarray]:
    """The committed golden master."""
    with np.load(gg.GOLDEN_PATH) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def actual() -> dict[str, np.ndarray]:
    """Freshly searched root statistics for every golden position."""
    return gg.compute_golden()


@pytest.fixture(scope="module")
def actual_with_snapshots() -> dict[str, np.ndarray]:
    """The same searches, with the power-of-two snapshot hook enabled."""
    return gg.compute_golden(snapshot_powers_of_two=True)


def test_position_set_unchanged(expected: dict[str, np.ndarray]) -> None:
    """The golden file must describe exactly the positions we search."""
    assert list(expected["__names__"]) == POSITION_NAMES


@pytest.mark.parametrize("name", POSITION_NAMES)
def test_visit_counts_are_identical(
    name: str, actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]
) -> None:
    """root.N must match the golden master exactly, element for element."""
    got = actual[f"{name}__N"]
    want = expected[f"{name}__N"]

    assert got.dtype == want.dtype
    if not np.array_equal(got, want):
        diff = np.flatnonzero(got != want)
        detail = ", ".join(
            f"a={int(i)}: {int(want[i])} -> {int(got[i])}" for i in diff[:10]
        )
        pytest.fail(
            f"{name}: root.N diverged at {diff.size} action(s) "
            f"({int(want.sum())} vs {int(got.sum())} total visits): {detail}"
        )


@pytest.mark.parametrize("name", POSITION_NAMES)
@pytest.mark.parametrize("field", ["W", "P"])
def test_float_stats_match(
    name: str,
    field: str,
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
) -> None:
    """root.W and root.P must match within rtol=0, atol=1e-9."""
    got = actual[f"{name}__{field}"]
    want = expected[f"{name}__{field}"]

    assert got.dtype == want.dtype
    assert np.allclose(got, want, rtol=0, atol=1e-9), (
        f"{name}: root.{field} max abs diff "
        f"{np.max(np.abs(got.astype(np.float64) - want.astype(np.float64)))}"
    )


@pytest.mark.parametrize("name", POSITION_NAMES)
def test_snapshots_leave_the_final_tree_identical(
    name: str,
    actual_with_snapshots: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
) -> None:
    """Recording snapshots mid-search must not perturb the search itself."""
    assert np.array_equal(actual_with_snapshots[f"{name}__N"], expected[f"{name}__N"])
    assert np.allclose(
        actual_with_snapshots[f"{name}__W"],
        expected[f"{name}__W"],
        rtol=0,
        atol=1e-9,
    )
