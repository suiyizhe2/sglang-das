"""CPU checks for grouped MQA planning and ragged/paged TopK coordinates.

Run directly; no accelerator runtime is needed.
"""

import importlib.util
import itertools
import runpy
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
# Load the registry without importing sglang and its accelerator dependencies.
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=1, suite="stage-a-test-cpu")

SPEC = importlib.util.spec_from_file_location(
    "mqa_request_split",
    ROOT / "python/sglang/srt/layers/attention/dsa/mqa_request_split.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
plan = MODULE.plan_mqa_request_slices
chunks = MODULE.iter_mqa_chunks


class TestMQARequestSplit(unittest.TestCase):
    def test_threshold_and_override(self):
        self.assertEqual(plan([896], [512]), ())
        self.assertEqual(
            plan([256] * 16, [512] * 16),
            (
                (0, 1280, 0, 2560),
                (1280, 2560, 2560, 5120),
                (2560, 4096, 5120, 8192),
            ),
        )
        self.assertEqual(len(plan([896] * 4, [21760] * 4)), 4)
        self.assertEqual(plan([1, 1], [1999999, 1]), ())
        self.assertEqual(len(plan([1, 1], [2000000, 1])), 2)
        self.assertEqual(len(plan([1, 1], [1, 1], 0)), 2)
        self.assertEqual(plan([896] * 4, [21760] * 4, 10**12), ())

    def test_empty_requests_preserve_offsets(self):
        self.assertEqual(
            plan([0, 3, 0, 2, 0], [5, 7, 9, 4, 8], 0),
            ((0, 3, 5, 12), (3, 5, 21, 25)),
        )
        self.assertEqual(plan([], []), ())
        with self.assertRaises(ValueError):
            plan([1], [], 0)
        with self.assertRaises(ValueError):
            plan([1], [1], -1)

    def test_threshold_is_per_extra_launch_not_full_matrix(self):
        # Full logits exceed two million cells, but K-prefix savings do not.
        self.assertEqual(plan([1000, 1], [1000, 100000]), ())
        # The global optimum is a partial split: keep requests 0-1 together and
        # launch request 2 separately.  Full per-request splitting would cost
        # two extra launches and is worse than this two-group partition.
        self.assertEqual(
            plan([1, 1, 1], [1000000, 1999999, 1]),
            ((0, 2, 0, 2999999), (2, 3, 2999999, 3000000)),
        )
        self.assertEqual(len(plan([1, 1, 1], [2000001, 2000001, 1])), 3)

    def test_grouped_partition_cost_beats_binary_choices(self):
        q_lens = [1, 1, 1]
        k_lens = [1000000, 1999999, 1]
        result = plan(q_lens, k_lens)

        def invalid_cells(groups):
            q_starts = [0, 1, 2]
            k_starts = [0, 1000000, 2999999]
            return sum(
                q_lens[i] * (k_starts[i] - group[2])
                for group in groups
                for i in range(len(q_lens))
                if group[0] <= q_starts[i] < group[1]
            )

        grouped_cost = invalid_cells(result) + 2000000 * (len(result) - 1)
        unsplit = ((0, 3, 0, 3000000),)
        fully_split = (
            (0, 1, 0, 1000000),
            (1, 2, 1000000, 2999999),
            (2, 3, 2999999, 3000000),
        )
        self.assertLess(grouped_cost, invalid_cells(unsplit))
        self.assertLess(
            grouped_cost,
            invalid_cells(fully_split) + 2000000 * (len(fully_split) - 1),
        )

    def test_dynamic_program_matches_exhaustive_partitions(self):
        rng = np.random.default_rng(11)
        for _ in range(100):
            request_count = int(rng.integers(2, 7))
            q_lens = rng.integers(1, 9, size=request_count).tolist()
            k_lens = rng.integers(1, 20, size=request_count).tolist()
            launch_cost = int(rng.integers(0, 100))

            q_starts = np.cumsum([0] + q_lens[:-1]).tolist()
            k_starts = np.cumsum([0] + k_lens[:-1]).tolist()

            def partition_from_cuts(cuts):
                groups = []
                start = 0
                for end in list(cuts) + [request_count]:
                    groups.append(
                        (
                            q_starts[start],
                            q_starts[end - 1] + q_lens[end - 1],
                            k_starts[start],
                            k_starts[end - 1] + k_lens[end - 1],
                        )
                    )
                    start = end
                return tuple(groups)

            def partition_cost(groups):
                invalid = sum(
                    q_lens[i] * (k_starts[i] - group[2])
                    for group in groups
                    for i in range(request_count)
                    if group[0] <= q_starts[i] < group[1]
                )
                return invalid + launch_cost * (len(groups) - 1)

            candidates = []
            boundaries = range(1, request_count)
            for cut_count in range(request_count):
                for cuts in itertools.combinations(boundaries, cut_count):
                    groups = partition_from_cuts(cuts)
                    candidates.append((partition_cost(groups), -len(groups)))
            expected_key = min(candidates)

            actual = plan(q_lens, k_lens, launch_cost)
            if not actual:
                actual = partition_from_cuts(())
            self.assertEqual((partition_cost(actual), -len(actual)), expected_key)

    def test_single_active_request_crops_zero_q_boundaries(self):
        self.assertEqual(
            plan([0, 3, 0], [5, 7, 9]),
            ((0, 3, 5, 12),),
        )

    def test_group_local_logits_keep_request_coordinates(self):
        q_lens = [1, 1, 1]
        k_lens = [10, 19, 1]
        groups = plan(q_lens, k_lens, 20)
        self.assertEqual(groups, ((0, 2, 0, 29), (2, 3, 29, 30)))

        rng = np.random.default_rng(19)
        q = rng.normal(size=(3, 2, 4))
        k = rng.normal(size=(30, 4))
        weights = rng.uniform(size=(3, 2))
        global_logits = (
            np.maximum(np.einsum("qhd,kd->qhk", q, k), 0) * weights[..., None]
        ).sum(axis=1)
        request_k_starts = [0, 10, 29]

        for q_start, q_end, group_k_start, group_k_end in groups:
            local_logits = (
                np.maximum(
                    np.einsum(
                        "qhd,kd->qhk",
                        q[q_start:q_end],
                        k[group_k_start:group_k_end],
                    ),
                    0,
                )
                * weights[q_start:q_end, :, None]
            ).sum(axis=1)
            for row in range(q_start, q_end):
                request_k_start = request_k_starts[row]
                request_k_end = request_k_start + k_lens[row]
                local_start = request_k_start - group_k_start
                local_end = request_k_end - group_k_start
                expected = np.argsort(
                    -global_logits[row, request_k_start:request_k_end]
                )[:4]
                actual = np.argsort(
                    -local_logits[row - q_start, local_start:local_end]
                )[:4]
                np.testing.assert_array_equal(
                    actual + request_k_start,
                    expected + request_k_start,
                )

    def test_budget_and_zero_budget_fast_path(self):
        slices = ((0, 5, 0, 10), (5, 8, 10, 30))
        self.assertEqual(
            list(chunks(slices, 80)),
            [
                (0, 2, 0, 10),
                (2, 4, 0, 10),
                (4, 5, 0, 10),
                (5, 6, 10, 30),
                (6, 7, 10, 30),
                (7, 8, 10, 30),
            ],
        )
        self.assertEqual(list(chunks(slices, 0)), list(slices))
        self.assertEqual(len(list(chunks(slices, 1))), 8)

    def test_logits_topk_and_cp_rank_local_lengths(self):
        rng = np.random.default_rng(7)
        # Different Q lengths also model the CPU lengths after CP sharding.
        q_lens, k_lens = [3, 2, 4], [9, 6, 12]
        q = rng.normal(size=(sum(q_lens), 3, 8))
        k = rng.normal(size=(sum(k_lens), 8))
        weights = rng.uniform(size=(sum(q_lens), 3))
        global_logits = (
            np.maximum(np.einsum("qhd,kd->qhk", q, k), 0) * weights[..., None]
        ).sum(axis=1)
        slices = plan(q_lens, k_lens, 0)
        page_table = rng.permutation(sum(k_lens)) + 100
        seen = []
        for start, end, k_start, k_end in chunks(slices, 96):
            local = (
                np.maximum(np.einsum("qhd,kd->qhk", q[start:end], k[k_start:k_end]), 0)
                * weights[start:end, :, None]
            ).sum(axis=1)
            np.testing.assert_allclose(local, global_logits[start:end, k_start:k_end])
            request = next(i for i, sl in enumerate(slices) if sl[0] <= start < sl[1])
            q_base = slices[request][0]
            for row in range(start, end):
                length = k_lens[request] - q_lens[request] + row - q_base + 1
                expected = np.argsort(-global_logits[row, k_start : k_start + length])[
                    :4
                ]
                actual = np.argsort(-local[row - start, :length])[:4]
                # Unfused output, ragged global offsets, and paged physical slots.
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(actual + k_start, expected + k_start)
                np.testing.assert_array_equal(
                    page_table[k_start + actual], page_table[k_start + expected]
                )
                seen.append(row)
        self.assertEqual(seen, list(range(sum(q_lens))))


if __name__ == "__main__":
    unittest.main()
