"""CPU control-flow regression using production methods without GPU imports.

This checks planning order/backend identity, not GPU attention numerics.
Usage: python scripts/validation/test_glm53_dcp_eager_metadata.py [SOURCE_ROOT]
"""
import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]


def methods(path, cls_name, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return type(cls_name, (), {n.name: namespace[n.name] for n in selected})


class MetadataRegression(unittest.TestCase):
    def setUp(self):
        ns = dict(contextlib=contextlib)
        self.active = SimpleNamespace(rows=None, plans=0)
        self.default = SimpleNamespace(rows=None, plans=0)
        self.pdmux = SimpleNamespace(rows=None, plans=0)
        for backend in (self.active, self.default, self.pdmux):
            def plan(fb, backend=backend):
                backend.rows = fb.batch_size
                backend.plans += 1
            backend.init_forward_metadata = plan
        ns.update(get_attn_backend=lambda: self.active,
                  has_forward_context=lambda: True,
                  is_cp_v2_active=lambda fb: False,
                  is_deepseek_dsa=lambda config: config.is_dsa,
                  get_req_to_token_pool=lambda: SimpleNamespace(req_to_token=None),
                  get_token_to_kv_pool=lambda: SimpleNamespace(get_kv_buffer_shape=lambda: [None]),
                  create_chunked_prefix_cache_kv_indices=None,
                  _is_hip=False,
                  ForwardContext=lambda **kw: SimpleNamespace(**kw),
                  forward_context=lambda ctx: contextlib.nullcontext(),
                  device_timer_ctx=lambda *a: contextlib.nullcontext())
        self.FB = methods('python/sglang/srt/model_executor/forward_batch_info.py',
                          'ForwardBatch', ['mark_forward_metadata_ready', 'needs_forward_metadata_init'], ns)
        self.Runner = methods('python/sglang/srt/model_executor/runner/eager_runner.py',
                              'EagerRunner', ['_resolve_decode_pdmux', '_execute_decode', '_execute_extend'], ns)

    def run_case(self, raw_bs, padded_bs, pdmux=False, context=True):
        fb = self.FB()
        fb.batch_size = raw_bs
        fb.input_ids = SimpleNamespace(shape=(raw_bs,))
        fb.positions = None
        self.active.init_forward_metadata(fb)
        fb.mark_forward_metadata_ready(replan_equivalent=True)
        # ModelRunner performs DP/MLP synchronization before eager dispatch.
        fb.batch_size = padded_bs
        fb.input_ids = SimpleNamespace(shape=(padded_bs,))
        used = self.pdmux if pdmux else (self.active if context else self.default)
        if used is not self.active:
            used.rows = raw_bs
        def forward(*args, **kwargs):
            self.assertEqual(used.rows, padded_bs,
                             'active attention metadata must match final query rows')
            return 'ok'
        runner = self.Runner()
        runner.enable_pdmux = pdmux
        runner.load_batch = lambda batch, _: batch
        runner.model_runner = SimpleNamespace(
            attn_backend=self.default, decode_attn_backend=self.pdmux,
            model=SimpleNamespace(forward=forward), _pp_kwargs=lambda _: {}, device_timer=None)
        runner._resolve_decode_pdmux.__func__.__globals__['has_forward_context'] = lambda: context
        self.assertEqual(runner._execute_decode(fb), 'ok')
        return used

    def test_uneven_dp_padding_replans_active_step(self):
        used = self.run_case(18, 20)
        self.assertEqual(used.plans, 2)
        self.assertEqual(self.default.plans, 0)

    def test_unchanged_shape_keeps_step_preplan(self):
        used = self.run_case(16, 16)
        self.assertEqual(used.plans, 1)

    def test_pdmux_preserves_stream_backend(self):
        used = self.run_case(18, 20, pdmux=True)
        self.assertEqual(used.plans, 1)
        self.assertEqual(self.default.plans, 0)

    def test_direct_call_falls_back_to_runner_backend(self):
        used = self.run_case(18, 20, context=False)
        self.assertEqual(used.plans, 1)

    def test_draft_worker_propagates_backend_replan_capability(self):
        for supported in (False, True):
            fb = self.FB()
            fb.batch_size = 13
            fb.input_ids = SimpleNamespace(shape=(13,))
            fb.forward_mode = SimpleNamespace(is_idle=lambda: False)
            ns = dict(contextlib=contextlib,
                      prepare_for_draft=lambda *args: (fb, False),
                      build_eagle_verify_input=lambda *args, **kwargs: 'verified')
            Worker = methods('python/sglang/srt/speculative/eagle_worker_v2.py',
                             'EagleDraftWorker', ['draft'], ns)
            worker = Worker()
            worker.req_to_token_pool = None
            worker.cuda_graph_runner = None
            worker.draft_runner = SimpleNamespace(canary_manager=None)
            worker.topk = 1
            worker.speculative_num_steps = 5
            worker.speculative_num_draft_tokens = 6
            worker.seed_dsa_topk_from_draft_extend = False
            worker.draft_attn_backend = SimpleNamespace(
                eager_metadata_replan_equivalent=supported,
                init_forward_metadata=lambda fb: None)
            worker.draft_forward = lambda fb: (None, None, None, None)
            worker.target_worker = None
            worker.tree_mask_mode = None
            worker.device = 'cpu'
            self.assertEqual(worker.draft(SimpleNamespace(spec_info=None)), 'verified')
            self.assertEqual(fb.forward_metadata_replan_equivalent, supported)

    def run_extend_case(self, is_dsa, verify):
        calls = []
        def planner(*args):
            self.assertIsNotNone(args[1], 'verify must not call the prefix-gather planner')
            calls.append('prefix-gather')
        fb = SimpleNamespace(
            forward_mode=SimpleNamespace(is_target_verify=lambda: verify),
            needs_forward_metadata_init=lambda: True,
            seq_lens=None, extend_prefix_lens=None if verify else [8],
            extend_prefix_lens_cpu=None if verify else [8],
            extend_seq_lens=[6], req_pool_indices=None, seq_lens_sum=14,
            input_ids=None, positions=None)
        runner = self.Runner()
        runner.enable_pdmux = False
        runner.load_batch = lambda fb, _: fb
        runner.model_runner = SimpleNamespace(
            ps=SimpleNamespace(attn_dcp_size=4),
            model_config=SimpleNamespace(hf_config=SimpleNamespace(is_dsa=is_dsa)),
            model=SimpleNamespace(prepare_context_parallel_metadata_for_dcp=planner,
                                  forward=lambda *a, **kw: 'ok'),
            attn_backend=SimpleNamespace(init_forward_metadata=lambda fb: calls.append('attn-plan')),
            _extend_forward_kwargs=lambda *a: {}, device_timer=None,
            kv_cache_dtype=None, device='cpu', prefill_cuda_graph_runner=None)
        self.assertEqual(runner._execute_extend(fb), 'ok')
        self.assertIn('attn-plan', calls)
        return calls

    def test_target_verify_uses_rank_local_kv(self):
        for is_dsa in (True, False):
            self.assertNotIn('prefix-gather', self.run_extend_case(is_dsa, True))

    def test_dsa_extend_uses_rank_local_kv(self):
        self.assertNotIn('prefix-gather', self.run_extend_case(True, False))

    def test_dense_extend_preserves_prefix_gather(self):
        self.assertIn('prefix-gather', self.run_extend_case(False, False))


if __name__ == '__main__':
    unittest.main()
