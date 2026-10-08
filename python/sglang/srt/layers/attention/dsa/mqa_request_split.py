"""CPU planning for grouped HCU ragged MQA launches.

Adapted from sglang-model commit 5ba8f17fdd44eca781f260593ca5bd4f72701085.
"""


def plan_mqa_request_slices(q_lens, k_lens, min_saved_cells_per_launch=2_000_000):
    """Partition consecutive requests to minimize invalid cells plus launches.

    ``min_saved_cells_per_launch`` converts one extra kernel launch to an
    equivalent number of logits cells.  A group pays for the invalid K-prefix
    cells between its first request and every later request in that group.  A
    dynamic program chooses the globally cheapest contiguous partition, so the
    result can be no split, a partial grouping, or one launch per request.

    Q chunking for the memory budget is applied separately by
    :func:`iter_mqa_chunks`.
    """
    if min_saved_cells_per_launch < 0:
        raise ValueError("MQA split threshold must be nonnegative")
    if len(q_lens) != len(k_lens):
        raise ValueError("MQA Q/K request counts differ")
    request_slices = []
    q_start = k_start = 0
    for q_len, k_len in zip(q_lens, k_lens):
        if q_len < 0 or k_len < 0:
            raise ValueError("MQA request lengths must be nonnegative")
        if q_len:
            request_slices.append(
                (q_start, q_start + q_len, k_start, k_start + k_len)
            )
        q_start += q_len
        k_start += k_len

    num_requests = len(request_slices)
    if num_requests == 0:
        return ()

    q_prefix = [0]
    qk_prefix = [0]
    for q_begin, q_end, request_k_start, _ in request_slices:
        q_len = q_end - q_begin
        q_prefix.append(q_prefix[-1] + q_len)
        qk_prefix.append(qk_prefix[-1] + q_len * request_k_start)

    # dp_cost[end] is the minimum cost for request_slices[:end].  On an exact
    # threshold tie, preserve the original MIN_SAVED_CELLS behavior and split.
    dp_cost = [0] + [None] * num_requests
    dp_groups = [0] + [None] * num_requests
    previous = [None] * (num_requests + 1)

    for end in range(1, num_requests + 1):
        best = None
        for start in range(end):
            group_k_start = request_slices[start][2]
            invalid_cells = (qk_prefix[end] - qk_prefix[start]) - group_k_start * (
                q_prefix[end] - q_prefix[start]
            )
            launch_cost = min_saved_cells_per_launch if start else 0
            candidate = (
                dp_cost[start] + invalid_cells + launch_cost,
                dp_groups[start] + 1,
            )
            if best is None or candidate[0] < best[0] or (
                candidate[0] == best[0] and candidate[1] > best[1]
            ):
                best = candidate
                previous[end] = start
        dp_cost[end], dp_groups[end] = best

    groups = []
    end = num_requests
    while end:
        start = previous[end]
        first = request_slices[start]
        last = request_slices[end - 1]
        groups.append((first[0], last[1], first[2], last[3]))
        end = start
    groups.reverse()

    # Keep the original single-launch fast path when the optimal group covers
    # the complete Q/K tensors.  A cropped single group is still useful when
    # zero-Q requests exist at either K boundary.
    full_slice = (0, q_start, 0, k_start)
    if len(groups) == 1 and groups[0] == full_slice:
        return ()
    return tuple(groups)


def iter_mqa_chunks(slices, budget_bytes, bytes_per_element=4):
    """Split each request group along Q while keeping only that group's K."""
    for q_start, q_end, k_start, k_end in slices:
        # A zero budget is the caller's small-matrix fast path: no chunk needed.
        max_rows = (
            max(1, budget_bytes // max((k_end - k_start) * bytes_per_element, 1))
            if budget_bytes
            else max(1, q_end - q_start)
        )
        for start in range(q_start, q_end, max_rows):
            yield start, min(start + max_rows, q_end), k_start, k_end
