# GLM-5.3 DCP eager metadata fix

This branch starts from `25720158d4beddc4586fcbf97f6f90f44b0b3974`, the
`feat/support_1M_context` source baseline used for the GLM-5.3 investigation.
It preserves that baseline's DCP/model support; it is not a port onto the
fork's older `main` branch.

## Failure and change

EAGLE can pre-plan draft attention metadata before DP/attention-TP padding.
When the batch exceeds the graph bucket and runs eagerly, padding can change
the query geometry while the ready marker retains the old page table. In the
observed failure, a draft batch of 18 became 20 rows: the page table stayed at
`(18, 62997)` while the query was `(20, 64, 512)`. The resulting top-k/page-table
row mismatch raised the assertion in `transform_index.py`.

The DSA multi-step backend now opts into equivalent replanning after padding,
and the draft worker propagates that capability. Eager execution plans against
the backend published by the active forward context, rather than the runner's
default draft-extend backend. The existing PDmux backend selection is retained.

A second eager failure occurred after the first fix: target verification used
the generic DCP prefix-KV gather planner with `extend_prefix_lens=None`. Eager
execution now skips this planner for target verification and DeepSeek DSA DCP
phases, matching their rank-local KV/gathered-Q execution path. Other dense
extend phases retain prefix gathering.

The page-table assertion remains enabled. The device regression retained
TP8/DP2/DCP4, `ag_rs`, graph maximum batch size 16 and maximum running requests
64; no concurrency reduction or larger graph bucket was used to avoid eager.

## Validation

Run the standalone CPU control-flow regression from this checkout:

```bash
python scripts/validation/test_glm53_dcp_eager_metadata.py
```

The eight checks exercise the production methods extracted through Python AST,
with small substitutes for GPU dependencies. They cover padded metadata,
active-step backend selection, unchanged geometry, PDmux, direct-call fallback,
worker capability propagation, verification/DSA local-KV handling, and ordinary
dense prefix gathering. They verify control flow, not GPU attention numerics
or complete framework integration. The same checks fail against the unmodified
baseline, including the padded-metadata and prefix-planner failure paths.

The original device regression on 2026-11-12 ran for 482.97 seconds with a
variable-length dataset and request concurrency 64: 206 requests completed,
zero client errors, and sampled eager batches reached 32. No repeat of the
assertion or verification-planner exception was observed in that bounded run.
It did not finish the entire dataset or validate a 1M-token request, long-run
stability, non-DCP compatibility, or GPU numerical equivalence.

The original device regression used request-level concurrency. A later
comparison workload uses session workers that send requests sequentially within
each session; it has not been claimed as the same completed regression.
Historical trace gaps are a separate investigation: this change has not been
shown to remove them.

## Additional container snapshot change

A separate commit preserves the container snapshot's DSA launch-validation
change: `dcp_comm_backend` accepts both `ag_rs` and `a2a`, instead of only
`ag_rs`. This is a validation-guard change; it does not add an A2A implementation.
The completed device regression above used `ag_rs`. A2A correctness,
compatibility and performance have not been established by that regression.
