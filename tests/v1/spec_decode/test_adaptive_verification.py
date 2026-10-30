# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.utils import multi_gpu_test
from vllm.config import (
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.worker.gpu.async_utils import RuntimeTimingCollector, StepTimingSample
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive_module
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    OnlineCostModel,
    assign_draft_token_budget,
    maybe_create_adaptive_verification_manager,
    resolve_adaptive_cudagraph_mode,
)
from vllm.v1.worker.gpu.states import RequestState


def make_manager(
    confidences: np.ndarray, verify_cost_ms: np.ndarray
) -> AdaptiveVerificationManager:
    num_reqs, num_steps = confidences.shape
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = num_steps
    manager._stale_confidences = [SimpleNamespace(np=confidences)]  # type: ignore[list-item]  # CPU-only confidence buffer.
    manager._stale_idx = 0
    manager.req_states = SimpleNamespace(
        req_id_to_index={"low": 0, "high": 1},
        num_computed_tokens_np=np.ones(num_reqs, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.ones(num_reqs, dtype=np.int32)),
    )
    manager.cost_tables = (np.zeros(num_reqs + 1), verify_cost_ms)
    manager.online_cost_model = None
    manager.synchronize_tp_decisions = False
    manager._max_total_logits = 1 << 30
    manager.num_bonus_tokens = 1
    return manager


@pytest.mark.parametrize(
    ("mode", "piecewise_capture_available", "expected"),
    [
        ("FULL_DECODE_ONLY", True, "FULL_DECODE_ONLY"),
        ("FULL_DECODE_ONLY", False, "FULL_DECODE_ONLY"),
        ("FULL", True, "FULL_AND_PIECEWISE"),
        ("FULL", False, "FULL_DECODE_ONLY"),
        ("FULL_AND_PIECEWISE", True, "FULL_AND_PIECEWISE"),
        ("FULL_AND_PIECEWISE", False, "FULL_DECODE_ONLY"),
    ],
)
def test_resolve_adaptive_cudagraph_mode(mode, piecewise_capture_available, expected):
    assert (
        resolve_adaptive_cudagraph_mode(
            CUDAGraphMode[mode], piecewise_capture_available=piecewise_capture_available
        )
        == CUDAGraphMode[expected]
    )


@pytest.mark.parametrize(
    "target_support,target_bound,device_offsets,additional,error",
    [
        pytest.param(AttentionCGSupport.ALWAYS, None, True, None, None, id="always"),
        pytest.param(
            AttentionCGSupport.UNIFORM_BATCH, 8, True, None, None, id="bounded"
        ),
        pytest.param(
            AttentionCGSupport.UNIFORM_BATCH,
            None,
            True,
            None,
            "TargetBackend allows none",
            id="unsupported",
        ),
        pytest.param(
            AttentionCGSupport.UNIFORM_BATCH,
            7,
            True,
            None,
            "up to 8, but TargetBackend allows at most 7",
            id="too-narrow",
        ),
        pytest.param(
            AttentionCGSupport.UNIFORM_BATCH,
            8,
            False,
            None,
            "trims verification requests",
            id="host-query-lens",
        ),
        pytest.param(
            AttentionCGSupport.ALWAYS,
            None,
            True,
            (AttentionCGSupport.UNIFORM_BATCH, "EncoderBackend"),
            "EncoderBackend allows none",
            id="additional-group",
        ),
    ],
)
def test_manager_checks_target_varlen_cudagraph_bound(
    monkeypatch, target_support, target_bound, device_offsets, additional, error
):
    """Target builders must replay varlen decode graphs of the verification
    width; draft-only groups are not checked."""

    def group(backend_name, layer_name, support, bound=None, mismatch=True):
        class Builder(AttentionMetadataBuilder):
            _cudagraph_support = support

            def __init__(self):
                pass

            def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
                raise NotImplementedError

            if bound is not None:

                @classmethod
                def get_varlen_cudagraph_max_query_len(cls, *_args):
                    return bound

        backend = type(
            backend_name,
            (),
            {"supports_device_cpu_query_lens_mismatch": staticmethod(lambda: mismatch)},
        )
        builder = Builder()
        return SimpleNamespace(
            layer_names=[layer_name],
            backend=backend,
            kv_cache_spec=None,
            get_metadata_builder=lambda _index: builder,
        )

    groups = [
        [
            group(
                "TargetBackend", "target", target_support, target_bound, device_offsets
            ),
            group("DraftBackend", "draft", AttentionCGSupport.UNIFORM_BATCH),
        ]
    ]
    created = object()
    monkeypatch.setattr(
        adaptive_module,
        "AdaptiveVerificationManager",
        lambda *_args, **_kwargs: created,
    )

    with pytest.raises(ValueError, match=error) if error else nullcontext():
        manager = maybe_create_adaptive_verification_manager(
            enable_adaptive_verification=True,
            attn_groups=groups,
            req_states=SimpleNamespace(num_speculative_steps=7),
            query_start_loc=object(),
            num_bonus_tokens=1,
            max_total_logits=1,
            vllm_config=VllmConfig(),
            target_layer_names={"target"},
            additional_attn_cg_support=additional,
        )
        assert manager is created


def test_budget_stops_where_marginal_drafts_stop_paying_for_themselves():
    # Verification is cheap up to two extra tokens, then jumps 100x; only the
    # highest-confidence draft is worth the cheap slot.
    manager = make_manager(
        np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0]),
    )

    manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    assert manager._batch_budget is not None
    valid_drafts, num_non_draft_tokens, draft_budget = manager._batch_budget

    assert draft_budget == 1
    assert valid_drafts == {"low": 2, "high": 2}
    assert num_non_draft_tokens == {"low": 1, "high": 1}


def test_profiled_batches_seed_cost_curves_via_consumer(monkeypatch):
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(max_num_batched_tokens=4096, max_num_reqs=64)
    manager.num_speculative_steps = 7
    manager.num_bonus_tokens = 1
    curves: dict[str, list[tuple[int, float]]] = {}
    monkeypatch.setattr(
        manager,
        "set_cost_curves",
        lambda draft, verify: curves.update(draft=draft, verify=verify),
    )

    timings = [
        StepTimingSample(
            forward_ms=float(batch["num_tokens"]),
            drafter_ms=1.0,
            num_target_tokens=batch["num_tokens"],
            num_reqs=batch["num_tokens"] // 8,
            # Only the captured sizes replay a graph; the tail sizes run eager.
            full_cudagraph=batch["num_tokens"] <= 1024,
        )
        for batch in manager.batches_to_profile([8, 1024])
    ]
    manager.set_initial_cost_curves(timings)

    # Tail beyond the last capture size: 1.5x then doubling to the max.
    assert curves["verify"] == [
        (8, 8.0),
        (1024, 1024.0),
        (1536, 1536.0),
        (2048, 2048.0),
        (4096, 4096.0),
    ]
    # Eager batches must not contribute to the draft curve: keyed by request
    # count they would land inside the captured range and, once made monotonic,
    # smear that eager cost across every larger request count.
    assert curves["draft"] == [(1, 1.0), (128, 1.0)]


def test_compact_batch_preserves_totals_and_bounds():
    # The CPU placeholder layout must keep the batch total equal to the GPU
    # total and every verification row within decode_query_len, or downstream
    # CPU metadata desyncs from the reallocated GPU boundaries.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
        np.array([1.0] * 44 + [100.0] * 3),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    num_tokens = manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40},
        {"low": [1, 2], "high": [3, 4]},
    )
    scheduled = np.array([3, 3, 40], dtype=np.int32)
    drafts = np.array([2, 2, 0], dtype=np.int32)
    cu_num_logits_np = np.array([0, 3, 6, 7], dtype=np.int32)
    compacted, _ = manager.compact_batch(drafts, scheduled, cu_num_logits_np)

    assert int(compacted.sum()) == num_tokens
    num_steps = manager.num_speculative_steps
    assert (compacted[:2] <= 1 + num_steps).all()
    assert compacted[2] == 40


def test_budget_caps_at_one_rejection_sampler_chunk():
    # The chunked verification path cannot address the compacted logits
    # layout, so the budget must keep total logits within a single chunk.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9]], dtype=np.float32),
        np.ones(7),
    )
    manager._max_total_logits = 3  # 2 bonus logits + at most 1 draft
    manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    assert manager._batch_budget is not None
    _, _, draft_budget = manager._batch_budget
    assert draft_budget <= 1


def test_zero_budget_rebuilds_cpu_cu_num_logits():
    # When one bonus row per request already overflows a verification chunk, the
    # budget clamps to zero but the batch still needs chunking. Every capacity is
    # zeroed on device, so the CPU can name that layout exactly -- and must, since
    # _iter_request_chunks slices the compacted logits with these offsets.
    #
    # The third request is a chunked prefill (no drafts, still mid-prompt). The
    # runner gives *every* request num_bonus_tokens logits rows regardless
    # (num_logits = num_draft_tokens_per_req + num_bonus_tokens), so the rebuilt
    # offsets stay uniform rather than skipping non-verification rows.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
        np.ones(64),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    manager._max_total_logits = 2  # < 3 requests * 1 bonus token

    manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40},
        {"low": [1, 2], "high": [3, 4]},
    )
    assert manager._batch_budget is not None
    _, _, draft_budget = manager._batch_budget
    assert draft_budget == 0

    scheduled = np.array([3, 3, 40], dtype=np.int32)
    drafts = np.array([2, 2, 0], dtype=np.int32)
    scheduled_cu_num_logits = np.array([0, 3, 6, 7], dtype=np.int32)
    compacted, cu_num_logits_np = manager.compact_batch(
        drafts, scheduled, scheduled_cu_num_logits
    )

    # One bonus row per request, matching cumsum(capacities + num_bonus_tokens)
    # with every capacity zeroed -- the prefill row included.
    expected = np.arange(4, dtype=np.int32) * manager.num_bonus_tokens
    assert np.array_equal(cu_num_logits_np, expected)
    assert cu_num_logits_np.dtype == scheduled_cu_num_logits.dtype
    # The prefill keeps its scheduled tokens; only drafts are dropped.
    assert np.array_equal(compacted, np.array([1, 1, 40], dtype=np.int32))


def global_topk_budget(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    capacities: torch.Tensor,
    draft_budget: int,
) -> torch.Tensor:
    survival = confidence_probs[idx_mapping].cumprod(dim=1)
    steps = torch.arange(survival.shape[1], device=survival.device)
    survival = survival.masked_fill(steps >= capacities[:, None], -float("inf"))
    admitted = torch.zeros(survival.numel(), dtype=torch.bool, device=survival.device)
    admitted[survival.flatten().topk(draft_budget).indices] = True
    return admitted.view_as(survival).sum(dim=1, dtype=torch.int32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_steps", [3, 8])
@pytest.mark.parametrize("num_reqs", [1, 2, 300])
def test_draft_budget_kernel_matches_global_topk(num_steps, num_reqs):
    """With distinct survivals the per-request admits equal a global top-k."""
    max_num_reqs = 512
    gen = torch.Generator(device="cuda").manual_seed(num_reqs * num_steps)
    confidence_probs = torch.rand(max_num_reqs, num_steps, device="cuda", generator=gen)
    idx_mapping = torch.randperm(max_num_reqs, device="cuda", generator=gen)
    idx_mapping = idx_mapping[:num_reqs]
    scheduled = torch.randint(
        1, num_steps + 1, (num_reqs,), dtype=torch.int32, device="cuda", generator=gen
    )
    scheduled[0] = num_steps
    survival = torch.empty(max_num_reqs * num_steps, dtype=torch.int32, device="cuda")
    num_scheduled = int(scheduled.sum())
    # The caller only invokes the kernel when the budget binds.
    budgets = {1, num_scheduled // 2, num_scheduled - 1}
    for draft_budget in sorted(b for b in budgets if 0 < b < num_scheduled):
        capacities = scheduled.clone()
        assign_draft_token_budget(
            confidence_probs, idx_mapping, capacities, draft_budget, survival
        )
        expected = global_topk_budget(
            confidence_probs, idx_mapping, scheduled, draft_budget
        )
        assert torch.equal(capacities, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_draft_budget_kernel_admits_exact_budget_on_ties():
    """Tied survivals are handed out in request order, across request blocks."""
    num_reqs, num_steps = 300, 3
    confidence_probs = torch.ones(num_reqs, num_steps, device="cuda")
    idx_mapping = torch.arange(num_reqs, device="cuda")
    capacities = torch.full((num_reqs,), num_steps, dtype=torch.int32, device="cuda")
    survival = torch.empty(num_reqs * num_steps, dtype=torch.int32, device="cuda")
    assign_draft_token_budget(confidence_probs, idx_mapping, capacities, 800, survival)
    assert capacities.tolist() == [3] * 266 + [2] + [0] * 33


def _online_cost_model() -> OnlineCostModel:
    # Startup prices are flat inside captured shapes and unpadded above 16.
    verify = np.array([0] + [4] * 4 + [8] * 4 + [16] * 8 + list(range(17, 33)))
    return OnlineCostModel(np.array([0.0, 2.0, 4.0]), verify.astype(float), [4, 8, 16])


def test_online_cost_feedback_waits_for_evidence_and_prices_padded_shapes():
    model = _online_cost_model()
    candidates = np.array([1, 4, 5, 8, 9, 16, 17])
    initial = np.array([6, 6, 10, 10, 18, 18, 19])
    samples = [StepTimingSample(16.0, 2.0, 8, 1, True)]

    model.update(samples * 9)
    assert model.num_samples == 9
    np.testing.assert_array_equal(model.get_costs(1, candidates), initial)
    model.update(samples)
    assert model.num_samples == 10
    np.testing.assert_array_equal(
        model.get_costs(1, candidates), [10, 10, 18, 18, 34, 34, 19]
    )
    # A request-count bucket without observations retains its startup draft cost.
    np.testing.assert_array_equal(model.get_costs(2, np.array([8, 17])), [20, 21])


def test_online_draft_cost_uses_five_samples_then_a_nine_sample_median():
    model = _online_cost_model()
    candidates = np.array([4, 17])
    high = StepTimingSample(4.0, 20.0, 4, 1, True)
    model.update([high] * 4)
    np.testing.assert_array_equal(model.get_costs(1, candidates), [6, 19])
    model.update([high])
    np.testing.assert_array_equal(model.get_costs(1, candidates), [24, 19])
    model.update([high] * 4)
    model.update([StepTimingSample(4.0, 2.0, 4, 1, True)] * 5)
    np.testing.assert_array_equal(model.get_costs(1, candidates), [6, 19])


def test_online_forward_cost_balances_shapes_and_forgets_old_observations():
    model = _online_cost_model()
    # Per-shape residual medians are 1, 3, and 6 ms/token. The hot shape must
    # not dominate the median, and one high outlier must not dominate shape 4.
    model.update(
        [StepTimingSample(8.0, 2.0, 4, 1, True)] * 9
        + [StepTimingSample(4000.0, 2.0, 4, 1, True)]
        + [StepTimingSample(32.0, 2.0, 8, 1, True)] * 10
        + [StepTimingSample(112.0, 2.0, 16, 1, True)] * 20
    )
    np.testing.assert_array_equal(
        model.get_costs(1, np.array([1, 5, 9, 17])), [18, 34, 66, 19]
    )
    # A full new window evicts all old shapes. Faster forwards never create
    # a negative correction to the startup price.
    model.update([StepTimingSample(4.0, 2.0, 8, 1, True)] * 5000)
    np.testing.assert_array_equal(
        model.get_costs(1, np.array([1, 5, 9, 17])), [6, 10, 18, 19]
    )


@pytest.mark.parametrize(
    ("forward_ms", "drafter_ms"),
    [(float("nan"), 20.0), (16.0, float("inf")), (0.0, 20.0), (16.0, -1.0)],
)
def test_online_cost_feedback_ignores_an_invalid_timing_entirely(
    forward_ms, drafter_ms
):
    model = _online_cost_model()
    model.update([StepTimingSample(16.0, 20.0, 8, 1, True)] * 4)
    model.update([StepTimingSample(forward_ms, drafter_ms, 8, 1, True)] * 10)
    model.update([StepTimingSample(16.0, 20.0, 7, 1, True)] * 10)
    model.update([StepTimingSample(16.0, 20.0, 8, 1, False)] * 10)
    np.testing.assert_array_equal(model.get_costs(1, np.array([8, 17])), [10, 19])


def test_online_cost_feedback_requires_adaptive_verification():
    with pytest.raises(ValueError, match="requires enable_adaptive_verification=True"):
        SpeculativeConfig(enable_online_cost_feedback=True)


def test_online_cost_feedback_accepts_tensor_and_data_parallel_configuration():
    config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=3,
        enable_online_cost_feedback=True,
        enable_adaptive_verification=True,
        target_parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            data_parallel_size=2,
            distributed_executor_backend="mp",
        ),
    )
    assert config.enable_online_cost_feedback
    assert config.target_parallel_config.tensor_parallel_size == 2
    assert config.target_parallel_config.data_parallel_size == 2


def _online_tp_dp_worker(rank: int, tp_size: int, store: str, feedback=True) -> None:
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="4")
    dp_rank = rank // tp_size
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp_size,
            data_parallel_size=4 // tp_size,
            data_parallel_rank=dp_rank,
            distributed_executor_backend="external_launcher",
        )
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=4,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"file://{store}",
            backend="nccl",
            timeout=timedelta(seconds=90),
        )
        initialize_model_parallel(tp_size)
        try:
            states = RequestState(2, 16, 8, 1, 32, device)
            for req_id in ("high", "low"):
                states.add_request(req_id, 1, [0], 1, 15)
            states.apply_staged_writes()
            buffers = InputBuffers(2, 8, device)
            batch = replace(
                InputBatch.make_dummy(2, 4, buffers), req_ids=["low", "high"]
            )
            manager = AdaptiveVerificationManager(
                states,
                buffers.query_start_loc,
                1,
                8,
                enable_online_cost_feedback=feedback,
            )
            list(manager.batches_to_profile([1, 2, 3, 4]))
            group = get_tp_group()
            up = float(np.nextafter(np.float32(0.5), np.float32(1)))
            down = float(np.nextafter(np.float32(0.5), np.float32(0)))

            def layout(stale, live, third_cost):
                manager.set_cost_curves(
                    [(1, 1.0), (2, 1.0)],
                    [(1, 1.0), (2, 1.0), (3, third_cost), (4, 100.0)],
                )
                for values in (stale, stale, live):
                    manager.record_confidences(
                        torch.tensor(values, device=device, dtype=torch.float32), batch
                    )
                total = manager.get_num_tokens(
                    {"low": 2, "high": 2}, {"low": [0], "high": [0]}
                )
                cu_logits, query_lens, *_ = manager.reallocate_drafts(
                    batch.req_ids, batch.idx_mapping
                )
                return total, cu_logits.tolist(), query_lens[:3].tolist()

            # One-ULP differences would choose different budgets without TP consensus.
            first = up if (dp_rank == 0) == group.is_first_rank else down
            budget_layout = layout([[first], [0.1]], [[first], [0.1]], 1.5)
            assert budget_layout[0] == (3 if dp_rank == 0 else 2)

            # Equal totals can still admit opposite requests on different TP ranks.
            live = (
                [[up], [down]]
                if (dp_rank == 0) == group.is_first_rank
                else [[down], [up]]
            )
            boundary_layout = layout([[0.5], [0.5]], live, 1.0)
            assert boundary_layout[2] == ([0, 2, 3] if dp_rank == 0 else [0, 1, 3])

            model = manager.online_cost_model
            assert (model is not None) == (feedback and group.is_first_rank)
            if model is not None:
                model.update(
                    [StepTimingSample(1 + 3 * (dp_rank + 1), 1, 3, 2, True)] * 10
                )
                assert model.num_samples == 10
            # Followers have no local calibration, yet must obey the learned budget.
            calibrated_total = manager.get_num_tokens(
                {"low": 2, "high": 2}, {"low": [0], "high": [0]}
            )
            assert calibrated_total == (2 if feedback else 3)
            layouts = [None] * 4
            dist.all_gather_object(
                layouts, (budget_layout, boundary_layout, calibrated_total)
            )
            for start in range(0, 4, tp_size):
                assert all(
                    row == layouts[start] for row in layouts[start : start + tp_size]
                )
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@multi_gpu_test(num_gpus=4)
def test_online_tp_dp_groups_agree_on_budget_and_request_boundaries(tp_size, tmp_path):
    """TP shares decisions while DP replicas retain independent cost feedback."""
    mp.spawn(_online_tp_dp_worker, args=(tp_size, str(tmp_path / "store")), nprocs=4)


@multi_gpu_test(num_gpus=4)
def test_online_tp_dp_consensus_remains_when_feedback_is_disabled(tmp_path):
    """Turning off calibration must not reintroduce divergent TP layouts."""
    mp.spawn(_online_tp_dp_worker, args=(2, str(tmp_path / "store"), False), nprocs=4)


@triton.jit
def _wait_for_runtime_timing_release(flag):
    while tl.atomic_add(flag, 0, sem="acquire") == 0:
        pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
def test_runtime_timing_keeps_pending_events_without_waiting_or_overwriting():
    collector = RuntimeTimingCollector()
    flag = torch.ones((), dtype=torch.int32, device="cuda")
    # Compile before holding the sampling stream: compilation may synchronize.
    _wait_for_runtime_timing_release[(1,)](flag, num_warps=1)
    torch.accelerator.synchronize()
    flag.zero_()
    initialized = torch.cuda.Event()
    initialized.record()
    sampling_stream = torch.cuda.Stream()
    release_stream = torch.cuda.Stream()
    sampling_stream.wait_event(initialized)
    release_stream.wait_event(initialized)
    queued = torch.cuda.Event()
    returned = Event()
    polled: list[list[StepTimingSample]] = []
    errors: list[Exception] = []

    def poll():
        try:
            polled.append(collector.collect_ready())
        except Exception as error:
            errors.append(error)
        finally:
            returned.set()

    thread = Thread(target=poll, daemon=True)
    try:
        with torch.cuda.stream(sampling_stream):
            _wait_for_runtime_timing_release[(1,)](flag, num_warps=1)
            for sample in range(1, 18):
                for _ in range(8):
                    collector.start(num_target_tokens=sample * 8, num_reqs=1)
                    collector.forward_end()
                    collector.drafter_start()
                    collector.drafter_end()
            queued.record()
        assert not queued.query()
        thread.start()
        returned_before_release = returned.wait(timeout=5)
    finally:
        with torch.cuda.stream(release_stream):
            flag.fill_(1)
        release_stream.synchronize()
        sampling_stream.synchronize()
        if thread.ident is not None:
            thread.join(timeout=5)

    assert returned_before_release, "collect_ready blocked on unfinished GPU work"
    assert not errors
    assert polled == [[]]
    completed = collector.collect_ready()
    assert [sample.num_target_tokens for sample in completed] == list(range(8, 129, 8))
    assert all(sample.num_reqs == 1 and sample.full_cudagraph for sample in completed)
    assert all(
        sample.forward_ms >= 0 and sample.drafter_ms >= 0 for sample in completed
    )
    assert collector.collect_ready() == []

    for _ in range(8):
        collector.start(num_target_tokens=8, num_reqs=1)
    collector.cancel()
    collector.forward_end()
    collector.drafter_start()
    collector.drafter_end()
    torch.accelerator.synchronize()
    assert collector.collect_ready() == []
