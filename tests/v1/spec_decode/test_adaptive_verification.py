# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import os
import weakref
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from statistics import median
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
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.worker.gpu.async_utils import (
    RuntimeTimingCollector,
    RuntimeTimingContext,
    RuntimeTimingSample,
    StepTimingSample,
)
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive_module
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    OnlineCostModel,
    assign_draft_token_budget,
    maybe_create_adaptive_verification_manager,
    resolve_adaptive_cudagraph_mode,
)
from vllm.v1.worker.gpu.spec_decode.cost_profile import DecodeCostProfiler
from vllm.v1.worker.gpu.spec_decode.parallel_cost import (
    BudgetProposal,
    select_parallel_budgets,
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
    manager.runtime_timing = None
    curves: dict[str, list[tuple[int, float]]] = {}
    monkeypatch.setattr(
        manager,
        "set_cost_curves",
        lambda draft, verify, online_forward_curve=None: curves.update(
            draft=draft, verify=verify
        ),
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
def test_online_profile_uses_trimmed_raw_forward_costs_without_changing_fallback(
    dist_init,
):
    device = torch.device("cuda")
    states = RequestState(2, 32, 16, 3, 32, device)
    buffers = InputBuffers(2, 16, device)
    manager = AdaptiveVerificationManager(
        states, buffers.query_start_loc, 1, 32, enable_online_cost_feedback=True
    )
    list(manager.batches_to_profile([4, 8]))
    samples = [
        StepTimingSample(value, 1.0, size, reqs, full)
        for size, reqs, full, values in [
            (4, 1, True, [1.0, 2.0, 3.0, 9.0, 100.0]),
            (8, 2, True, [1.0, 1.5, 2.0, 2.5, 3.0]),
            (16, 2, False, [9.0] * 5),
        ]
        for value in values
    ]
    manager.set_initial_cost_curves(samples, online_samples=samples)
    assert manager.cost_tables is not None
    np.testing.assert_array_equal(manager.cost_tables[1][[4, 8, 16]], [3, 3, 9])
    model = manager.online_cost_model
    assert model is not None
    candidates = np.array([1, 4, 5, 8, 9])
    np.testing.assert_array_equal(
        model.get_costs(1, candidates), [14 / 3, 14 / 3, 2, 2, 10]
    )
    # Residuals are measured against raw 2ms, not the monotonic fallback's 3ms.
    model.update([RuntimeTimingSample(4.0, 1.0, 8, 2)] * 10)
    np.testing.assert_array_equal(
        model.get_costs(1, candidates), [20 / 3, 20 / 3, 5, 5, 10]
    )


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
    return OnlineCostModel(
        np.array([0.0, 2.0, 4.0, 6.0]), verify.astype(float), [4, 8, 16]
    )


def test_online_cost_feedback_waits_for_evidence_and_prices_padded_shapes():
    model = _online_cost_model()
    candidates = np.array([1, 4, 5, 8, 9, 16, 17])
    np.testing.assert_array_equal(
        model.get_costs(1, candidates), [4, 4, 8, 8, 16, 16, 19]
    )
    samples = [RuntimeTimingSample(16.0, 2.0, 8, 1)]

    model.update(samples * 9)
    assert model.num_samples == 9
    np.testing.assert_array_equal(
        model.get_costs(1, candidates), [6, 6, 10, 10, 18, 18, 19]
    )
    model.update(samples)
    assert model.num_samples == 10
    np.testing.assert_array_equal(
        model.get_costs(1, candidates), [10, 10, 18, 18, 34, 34, 19]
    )
    # Unseen request counts use global sampler observations for FULL graphs,
    # while eager candidates retain their original startup draft cost.
    np.testing.assert_array_equal(model.get_costs(2, np.array([8, 17])), [18, 21])


def test_online_sampler_cost_uses_bucket_global_and_zero_fallbacks():
    model = _online_cost_model()
    candidates = np.array([4, 17])
    high = RuntimeTimingSample(4.0, 100.0, 4, 1)
    low = RuntimeTimingSample(4.0, 10.0, 4, 2)
    model.update([high] * 4)
    np.testing.assert_array_equal(model.get_costs(1, candidates), [4, 19])
    model.update([low])
    np.testing.assert_array_equal(model.get_costs(3, candidates), [104, 23])
    model.update([low] * 3)
    np.testing.assert_array_equal(model.get_costs(1, candidates), [59, 19])
    model.update([high])
    model.update([low] * 4)
    np.testing.assert_array_equal(model.get_costs(1, candidates), [104, 19])
    np.testing.assert_array_equal(model.get_costs(2, candidates), [14, 21])
    np.testing.assert_array_equal(model.get_costs(3, candidates), [14, 23])
    # Five new observations displace four of the old bucket's nine entries.
    # The unseen bucket also follows the global nine-entry window.
    model.update([RuntimeTimingSample(4.0, 1.0, 4, 1)] * 5)
    np.testing.assert_array_equal(model.get_costs(1, candidates), [5, 19])
    np.testing.assert_array_equal(model.get_costs(2, candidates), [14, 21])
    np.testing.assert_array_equal(model.get_costs(3, candidates), [5, 23])


def test_online_forward_cost_balances_shapes_and_forgets_old_observations():
    model = _online_cost_model()
    # Per-shape residual medians are 1, 3, and 6 ms/token. The hot shape must
    # not dominate the median, and one high outlier must not dominate shape 4.
    model.update(
        [RuntimeTimingSample(8.0, 2.0, 4, 1)] * 9
        + [RuntimeTimingSample(4000.0, 2.0, 4, 1)]
        + [RuntimeTimingSample(32.0, 2.0, 8, 1)] * 10
        + [RuntimeTimingSample(112.0, 2.0, 16, 1)] * 20
    )
    np.testing.assert_array_equal(
        model.get_costs(1, np.array([1, 5, 9, 17])), [18, 34, 66, 19]
    )
    # A full new window evicts all old shapes. Faster forwards never create
    # a negative correction to the startup price.
    model.update([RuntimeTimingSample(4.0, 2.0, 8, 1)] * 5000)
    np.testing.assert_array_equal(
        model.get_costs(1, np.array([1, 5, 9, 17])), [6, 10, 18, 19]
    )


@pytest.mark.parametrize(
    ("forward_ms", "sampler_ms"),
    [(float("nan"), 20.0), (16.0, float("inf")), (0.0, 20.0), (16.0, -1.0)],
)
def test_online_cost_feedback_ignores_an_invalid_timing_entirely(
    forward_ms, sampler_ms
):
    model = _online_cost_model()
    model.update([RuntimeTimingSample(16.0, 20.0, 8, 1)] * 4)
    model.update([RuntimeTimingSample(forward_ms, sampler_ms, 8, 1)] * 10)
    model.update([RuntimeTimingSample(16.0, 20.0, 7, 1)] * 10)
    model.update([RuntimeTimingSample(16.0, 20.0, 8, 0)] * 10)
    assert model.num_samples == 4
    np.testing.assert_array_equal(model.get_costs(1, np.array([8, 17])), [8, 19])


def _reference_online_costs(history, num_reqs, candidates):
    """Recompute from complete histories, without incremental median state."""
    forward: dict[int, list[float]] = {}
    for sample in history[-5000:]:
        size = sample.num_target_tokens
        forward.setdefault(size, []).append((sample.forward_ms - size) / size)
    shape_medians = [median(values) for values in forward.values() if len(values) >= 10]
    slope = max(0.0, median(shape_medians)) if shape_medians else 0.0
    local = [sample.sampler_ms for sample in history if sample.num_reqs == num_reqs][
        -9:
    ]
    global_samples = [sample.sampler_ms for sample in history][-9:]
    sampler = (
        median(local)
        if len(local) >= 5
        else median(global_samples)
        if len(global_samples) >= 5
        else 0.0
    )
    expected = []
    for tokens in candidates:
        if tokens > 16:
            expected.append([0, 2, 4, 6][num_reqs] + tokens)
        else:
            padded = next(size for size in [4, 8, 16] if tokens <= size)
            expected.append(padded + slope * padded + sampler)
    return np.array(expected)


@pytest.mark.parametrize("unique_residuals", [False, True])
def test_incremental_cost_medians_match_recomputation_across_evictions(
    unique_residuals,
):
    model = _online_cost_model()
    observations = []
    for index in range(5040):
        size = [4, 8, 16][(index // 7) % 3]
        residual = (
            index / 8192 - 0.5
            if unique_residuals
            else [-0.5, 0.0, 0.5, 1.0, 1.0][index % 5]
        )
        observations.append(
            RuntimeTimingSample(
                size * (1 + residual),
                [1.0, 1.0, 3.0, 9.0][index % 4],
                size,
                1 + index % 3,
            )
        )
    candidates = np.array([1, 4, 5, 8, 9, 16, 17])
    checkpoints = [
        1,
        4,
        5,
        9,
        10,
        11,
        20,
        29,
        40,
        50,
        100,
        4999,
        5000,
        5001,
        5010,
        5040,
    ]
    previous = 0
    for count in checkpoints:
        model.update(observations[previous:count])
        previous = count
        assert model.num_samples == min(count, 5000)
        for num_reqs in [1, 2, 3]:
            np.testing.assert_array_equal(
                model.get_costs(num_reqs, candidates),
                _reference_online_costs(observations[:count], num_reqs, candidates),
                err_msg=f"after {count} samples, request bucket {num_reqs}",
            )


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


def test_parallel_budget_accounts_for_idle_padding_and_group_sampler():
    model = OnlineCostModel(
        np.zeros(9), np.ones(9), [2, 4], forward_curve=[(2, 1.0), (4, 4.0)]
    )
    proposals = [
        BudgetProposal(0, 4, np.zeros(1)),
        BudgetProposal(1, 1, np.array([1.0, 1.9, 2.8, 3.7])),
    ]
    # The idle replica already forces four physical tokens. Trimming the active
    # replica to two cannot buy a cheaper graph for the cooperating forward.
    prices = np.zeros(9)
    prices[model.graph_sizes] = model.get_forward_costs()
    budgets, physical, cost = select_parallel_budgets(
        proposals, prices, np.array(model.get_sampler_costs())
    )
    assert budgets == (0, 3)
    assert physical == 4
    assert cost == 4.0
    model.update(
        [
            RuntimeTimingSample(4.0, sampler_ms, 4, reqs)
            for reqs, sampler_ms in [(1, 1.0), (3, 7.0)]
            for _ in range(10)
        ]
    )
    prices[model.graph_sizes] = model.get_forward_costs()
    proposals = [BudgetProposal(1, 1, np.ones(1)), BudgetProposal(3, 3, np.ones(1))]
    assert (
        select_parallel_budgets(proposals, prices, np.array(model.get_sampler_costs()))[
            2
        ]
        == 11.0
    )


def test_coupled_sampler_uses_agreed_draft_batch_count():
    """The driver's local N cannot identify a drafter padded by another DP."""
    model = OnlineCostModel(np.zeros(5), np.ones(9), [4], forward_curve=[(4, 4.0)])

    def sample(n_across_dp, sampler_ms):
        context = RuntimeTimingContext(
            1,
            1,
            0,
            n_across_dp,
            (4, 4),
            n_across_dp,
            0.0,
        )
        return RuntimeTimingSample(4.0, sampler_ms, 4, 1, context)

    model.update([sample((1, 3), 7.0)] * 5)
    model.update([sample((1, 1), 1.0)] * 5)
    assert model.get_costs(3, np.array([4])).tolist() == [11.0]
    assert model.get_costs(1, np.array([4])).tolist() == [5.0]


def test_online_prices_exclude_piecewise_only_profile_shapes(dist_init):
    """A native mixed graph profile may contain sizes without a FULL replay."""
    device = torch.device("cuda")
    states = RequestState(2, 32, 16, 3, 32, device)
    buffers = InputBuffers(2, 16, device)
    manager = AdaptiveVerificationManager(
        states, buffers.query_start_loc, 1, 32, enable_online_cost_feedback=True
    )
    list(manager.batches_to_profile([4, 8]))
    manager.set_initial_cost_curves(
        [
            StepTimingSample(4.0, 1.0, 4, 1, True),
        ]
        * 5
        + [StepTimingSample(8.0, 1.0, 8, 2, False)] * 5,
        online_samples=[StepTimingSample(4.0, 0.0, 4, 1, True)] * 5,
    )
    assert manager.online_cost_model is not None
    assert manager.online_cost_model.graph_sizes.tolist() == [4]
    states.add_request("req", 1, [0], 1, 15)
    states.apply_staged_writes()
    assert manager.get_num_tokens({"req": 4}, {"req": [1, 2, 3]}) <= 4


def test_online_baseline_does_not_replace_native_profile_prices(dist_init, monkeypatch):
    """The no-history forward baseline must not alter native fallback tables."""
    monkeypatch.setenv("VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", "64")
    device = torch.device("cuda")
    states = RequestState(2, 32, 16, 3, 32, device)
    buffers = InputBuffers(2, 16, device)
    manager = AdaptiveVerificationManager(
        states, buffers.query_start_loc, 1, 32, enable_online_cost_feedback=True
    )
    batches = list(manager.batches_to_profile([4, 8]))
    samples = [
        StepTimingSample(
            9.0,
            1.0,
            b["num_tokens"],
            1 if b["num_tokens"] == 4 else 2,
            b["num_tokens"] <= 8,
        )
        for b in batches
    ]
    manager.set_initial_cost_curves(
        samples,
        online_samples=[
            StepTimingSample(1.0, 99.0, size, reqs, True)
            for size, reqs in [(8, 2), (4, 1)]
            for _ in range(5)
        ],
    )
    assert manager.cost_tables is not None
    assert manager.cost_tables[0][1] == 1.0
    assert manager.cost_tables[1][4] == 9.0
    assert manager.online_cost_model is not None
    assert manager.online_cost_model.get_costs(1, np.array([4])).tolist() == [1.0]


def test_online_moe_without_dp_uses_tp_driver(dist_init):
    """One logical batch needs TP consensus, even when experts cooperate."""
    device = torch.device("cuda")
    states = RequestState(2, 32, 16, 3, 32, device)
    buffers = InputBuffers(2, 16, device)
    manager = AdaptiveVerificationManager(
        states,
        buffers.query_start_loc,
        1,
        32,
        enable_online_cost_feedback=True,
        coordinate_moe_costs=True,
    )
    assert manager.parallel_costs is None
    list(manager.batches_to_profile([4, 8]))
    manager.set_cost_curves([(1, 1.0), (2, 1.0)], [(4, 4.0), (8, 8.0)])
    assert manager.online_cost_model is not None
    manager.online_cost_model.update([RuntimeTimingSample(8.0, 2.0, 4, 1)] * 10)
    assert manager.online_cost_model.get_costs(1, np.array([4])).tolist() == [10.0]


def _small_moe_model_config(tmp_path, max_model_len=16):
    from transformers import Qwen3MoeConfig

    from vllm.config import ModelConfig

    Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=64,
        max_position_embeddings=16,
    ).save_pretrained(tmp_path)
    return ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
    )


@pytest.mark.parametrize(
    "dp_size,backend,dynamic,error",
    [
        (1, "deepep_high_throughput", False, None),
        (1, "allgather_reducescatter", True, None),
        (2, "deepep_high_throughput", False, "requires static groups"),
        (2, "allgather_reducescatter", True, None),
    ],
)
def test_online_moe_factory_limits_joint_coordination_to_dp(
    dist_init, tmp_path, dp_size, backend, dynamic, error
):
    """DP=1 keeps the native schedule/backend while DP>1 guards joint budgets."""
    model_config = _small_moe_model_config(tmp_path)
    schedule = [(1, 2, 1), (3, 4, 3)] if dynamic else None
    config = VllmConfig(
        model_config=model_config,
        parallel_config=ParallelConfig(
            data_parallel_size=dp_size,
            enable_expert_parallel=True,
            all2all_backend=backend,
            distributed_executor_backend="mp",
        ),
        speculative_config=SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=3,
            enable_adaptive_verification=True,
            enable_online_cost_feedback=True,
            num_speculative_tokens_per_batch_size=schedule,
        ),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_num_seqs=2,
            max_num_batched_tokens=16,
            max_model_len=16,
        ),
    )
    assert config.speculative_config is not None
    if dp_size > 1 and dynamic:
        # Native config resolves this before the AV factory: retain that
        # fallback instead of restoring an unsupported dynamic schedule.
        assert config.speculative_config.num_speculative_tokens_per_batch_size is None
        assert config.speculative_config.num_speculative_tokens == 3
        assert not config.speculative_config.uses_dynamic_speculative_decoding()
        return
    device = torch.device("cuda")
    states = RequestState(2, 16, 16, 3, 64, device)
    buffers = InputBuffers(2, 16, device)
    kwargs = dict(
        enable_adaptive_verification=True,
        attn_groups=[],
        req_states=states,
        query_start_loc=buffers.query_start_loc,
        num_bonus_tokens=1,
        max_total_logits=16,
        vllm_config=config,
    )
    if error:
        with pytest.raises(ValueError, match=error):
            maybe_create_adaptive_verification_manager(**kwargs)
        return

    manager = maybe_create_adaptive_verification_manager(**kwargs)
    assert manager is not None and manager.parallel_costs is None
    assert config.parallel_config.all2all_backend == backend
    assert config.speculative_config.num_speculative_tokens_per_batch_size == schedule
    list(manager.batches_to_profile([4, 8]))
    manager.set_cost_curves([(1, 1.0), (2, 1.0)], [(4, 4.0), (8, 8.0)])
    assert manager.online_cost_model is not None
    manager.online_cost_model.update([RuntimeTimingSample(8.0, 2.0, 4, 1)] * 10)
    assert manager.online_cost_model.get_costs(1, np.array([4])).tolist() == [10.0]
    # Removing coordination-only guards must not bypass native graph capability.
    with pytest.raises(ValueError, match="NativeBackend allows none"):
        maybe_create_adaptive_verification_manager(
            **kwargs,
            additional_attn_cg_support=(
                AttentionCGSupport.UNIFORM_BATCH,
                "NativeBackend",
            ),
        )


@pytest.mark.parametrize("max_num_reqs", [3, 128])
def test_sampler_prices_refresh_after_new_observations(max_num_reqs):
    """Published sampler prices must track both bucket and fallback updates."""
    model = OnlineCostModel(np.zeros(max_num_reqs + 1), np.ones(513), [512])
    assert model.get_sampler_costs() == [0.0] * (max_num_reqs + 1)
    model.update([RuntimeTimingSample(1.0, 7.0, 512, 2)] * 5)
    assert model.get_sampler_costs() == [0.0] + [7.0] * max_num_reqs
    model.update([RuntimeTimingSample(1.0, 1.0, 512, 1)] * 5)
    assert model.get_sampler_costs() == [0.0, 1.0, 7.0] + [1.0] * (max_num_reqs - 2)
    observations = [(2, latency) for latency in range(9, 18)] + [
        (max_num_reqs, latency) for latency in range(23, 28)
    ]
    for num_reqs, latency in observations:
        model.update([RuntimeTimingSample(1.0, float(latency), 512, num_reqs)])
        costs = model.get_sampler_costs()
        assert costs == [0.0] + [
            model.get_costs(n, np.array([512]))[0] - 1.0
            for n in range(1, max_num_reqs + 1)
        ]
        assert model.get_sampler_costs() is costs
    model.update([RuntimeTimingSample(float("nan"), 999.0, 512, 2)])
    assert model.get_sampler_costs() is costs


def test_shorter_proposal_uses_native_prices_without_changing_schedule(dist_init):
    """Full-step sampler observations cannot price a shorter current proposal."""
    device = torch.device("cuda")
    states = RequestState(1, 16, 8, 3, 64, device)
    states.add_request("req", 1, [0], 1, 15)
    states.apply_staged_writes()
    buffers = InputBuffers(1, 8, device)
    manager = AdaptiveVerificationManager(
        states, buffers.query_start_loc, 1, 8, enable_online_cost_feedback=True
    )
    manager.add_request(states.req_id_to_index["req"])
    list(manager.batches_to_profile([1, 2, 3]))
    manager.set_cost_curves([(1, 1.0)], [(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)])
    batch = InputBatch.make_dummy(1, 2, buffers)
    for _ in range(3):
        manager.record_confidences(torch.full((1, 3), 0.1, device=device), batch)
    assert manager.online_cost_model is not None
    manager.online_cost_model.update([RuntimeTimingSample(31.0, 1.0, 3, 1)] * 10)
    scheduled, drafts = {"req": 2}, {"req": [1]}
    assert manager.get_num_tokens(scheduled, drafts, num_proposal_steps=3) == 1
    assert manager.get_num_tokens(scheduled, drafts, num_proposal_steps=1) == 2
    assert scheduled == {"req": 2} and drafts == {"req": [1]}
    assert manager.online_cost_model.num_samples == 10


@torch.inference_mode()
@pytest.mark.parametrize("routing", ["plain", "bias", "hash"])
def test_startup_random_routing_preserves_serving_graphs_and_rng(
    dist_init, routing, tmp_path
):
    """Only temporary graphs may route away from a deliberately biased gate."""
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device("cuda")
    config = VllmConfig(
        model_config=_small_moe_model_config(tmp_path, max_model_len=8),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_num_seqs=8,
            max_num_batched_tokens=8,
            max_model_len=8,
        ),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[4, 8],
            max_cudagraph_capture_size=8,
        ),
    )
    config.kernel_config.moe_backend = "triton"
    with set_current_vllm_config(config), torch.device(device):
        init_workspace_manager(torch.accelerator.current_device_index())
        model = FusedMoEFactory(
            num_experts=8,
            top_k=1,
            hidden_size=128,
            intermediate_size=128,
            params_dtype=torch.bfloat16,
            renormalize=True,
            scoring_func="sqrtsoftplus" if routing == "hash" else "softmax",
            e_score_correction_bias=(
                torch.nn.Parameter(torch.tensor([1000.0] + [0.0] * 7, device=device))
                if routing in ("bias", "hash")
                else None
            ),
            hash_indices_table=(
                torch.zeros(16, 1, dtype=torch.int64, device=device)
                if routing == "hash"
                else None
            ),
            prefix="startup_random_routing",
        )
        model.routed_experts.w13_weight.normal_(0, 0.01)
        model.routed_experts.w2_weight.normal_(0, 0.01)
        model._quant_method.process_weights_after_loading(model.routed_experts)
        buffers = InputBuffers(8, 8, device)
        inputs = torch.ones(8, 128, dtype=torch.bfloat16, device=device)
        logits = torch.full((8, 8), -20.0, device=device)
        logits[:, 0] = 20.0
        output = torch.empty_like(inputs)
        graphs = CudaGraphManager(config, device, CUDAGraphMode.FULL_DECODE_ONLY, 1)
        profiler = DecodeCostProfiler(model, buffers, config, moe=True)
        measuring = False

        def create_forward(desc, warmup):
            buffers.is_padding.fill_(True)
            buffers.input_ids.zero_()

            def forward(mode):
                assert not measuring, "Measurement must not call the Python model"
                with set_forward_context(
                    None,
                    config,
                    num_tokens=desc.num_tokens,
                    is_padding=buffers.is_padding[: desc.num_tokens],
                ):
                    output[: desc.num_tokens].copy_(
                        model(
                            inputs[: desc.num_tokens],
                            logits[: desc.num_tokens],
                            input_ids=buffers.input_ids[: desc.num_tokens],
                        )
                    )

            return forward

        before = torch.cuda.get_rng_state(device)
        graphs.capture(create_forward, capture_for_profile=profiler.capture)
        assert profiler.samples == []
        assert model.profile_random_routing is None
        measuring = True
        profiler.measure()
        measuring = False
        assert [s.num_target_tokens for s in profiler.samples] == [8] * 5 + [4] * 5
        assert all(s.forward_ms > 0 and s.drafter_ms == 0 for s in profiler.samples)
        weights, ids = profiler.routes[model]
        assert torch.unique(ids[:8]).numel() > 1
        # Compare the measured graph's output with the native expert entry fed
        # explicit physical selections, independently of the bias/hash router.
        measured_output = output[:4].clone()
        expected_weights, expected_ids = weights[:4], ids[:4]
        with set_forward_context(None, config, num_tokens=4):
            reference = model.routed_experts.forward_modular(
                inputs[:4], expected_weights, expected_ids, None, None
            )
        torch.testing.assert_close(measured_output, reference, rtol=0, atol=0)
        assert model.profile_random_routing is None
        assert torch.equal(torch.cuda.get_rng_state(device), before)

        desc = graphs.dispatch(8, 8, 1, 0)
        forward = create_forward(desc, warmup=False)
        buffers.is_padding.fill_(False)
        forward(CUDAGraphMode.NONE)
        expected = output.clone()
        graphs.run_fullgraph(desc)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

        def failing_forward(desc, warmup):
            raise RuntimeError("profiling preparation failed")

        with pytest.raises(RuntimeError, match="profiling preparation failed"):
            profiler.capture(desc, failing_forward)
        assert model.profile_random_routing is None
        assert torch.equal(torch.cuda.get_rng_state(device), before)

        owner = weakref.ref(profiler)
        del profiler
        gc.collect()
        assert owner() is not None, "Serving graphs must retain startup capture owners"
        graphs.run_fullgraph(desc)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        graphs.release_graphs()
        gc.collect()
        assert owner() is None


@multi_gpu_test(num_gpus=8)
def test_online_tp_dp_metadata_only_followers_do_not_wait_for_confidence_copies(
    tmp_path,
):
    """Only TP leaders consume host confidences for the joint DP budget."""
    mp.spawn(
        _metadata_only_follower_worker,
        args=(str(tmp_path / "metadata-only-store"),),
        nprocs=8,
        join=True,
    )


def _metadata_only_follower_worker(rank: int, store: str) -> None:
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="8")
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=4,
            data_parallel_size=2,
            data_parallel_rank=rank // 4,
            enable_expert_parallel=True,
            distributed_executor_backend="external_launcher",
        )
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=8,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"file://{store}",
            backend="nccl",
            timeout=timedelta(seconds=90),
        )
        initialize_model_parallel(4)
        try:
            states = RequestState(2, 16, 8, 3, 32, device)
            req_ids = ["first", "second"]
            for req_id in req_ids:
                states.add_request(req_id, 1, [0], 1, 15)
            states.apply_staged_writes()
            buffers = InputBuffers(2, 8, device)
            slots = np.array(
                [states.req_id_to_index[r] for r in req_ids], dtype=np.intp
            )
            batch = replace(
                InputBatch.make_dummy(2, 8, buffers, uniform_token_count=4),
                req_ids=req_ids,
                idx_mapping=torch.from_numpy(slots).to(device),
                idx_mapping_np=slots,
            )
            manager = AdaptiveVerificationManager(
                states,
                buffers.query_start_loc,
                1,
                8,
                enable_online_cost_feedback=True,
                coordinate_moe_costs=True,
            )
            probabilities = torch.full((2, 3), 0.9, device=device)
            for _ in range(2):
                manager.record_confidences(probabilities, batch)
            torch.accelerator.synchronize()
            follower = not get_tp_group().is_first_rank
            done = torch.Event(device=device)
            if follower:
                # Bounded GPU work keeps the old copy pending; query, not a
                # wall-time threshold, proves that the host returned early.
                torch.cuda._sleep(2_000_000_000)
                done.record()
            for _ in range(2):
                manager.record_confidences(probabilities, batch)
            blocked = torch.tensor(int(follower and done.query()))
            torch.accelerator.synchronize()
            dist.all_reduce(blocked, group=get_world_group().cpu_group)
            assert blocked.item() == 0, "Metadata-only followers waited for D2H"

            list(manager.batches_to_profile([4, 8]))
            manager.set_initial_cost_curves(
                [StepTimingSample(1.0, 1.0, 4, 1, True)] * 5
                + [StepTimingSample(100.0, 1.0, 8, 2, True)] * 5,
                online_samples=[
                    StepTimingSample(1.0, 0.0, size, 1, True)
                    for size in (4, 8)
                    for _ in range(5)
                ],
            )
            tokens = dict.fromkeys(req_ids, 4)
            drafts = dict.fromkeys(req_ids, [0, 0, 0])
            coordinator = manager.parallel_costs
            assert coordinator is not None
            for phase, (profiling, prefill) in enumerate(
                ((False, False), (rank == 1, False), (False, rank >= 4))
            ):
                manager.coordinate_parallel_step(
                    tokens,
                    drafts,
                    has_prefill=prefill,
                    dummy_run=False,
                    profiling=profiling,
                )
                assert coordinator.decision is not None
                if phase == 0:
                    assert coordinator.decision.budgets == (6, 6)
                    agreed = manager.get_dp_sync_state()
                    assert agreed is not None
                    assert agreed.num_tokens_across_dp.tolist() == [8, 8]
                    assert agreed.uniform_token_count == agreed.max_query_len == 4
                    assert agreed.num_reqs == 2 and not agreed.eager
                    assert not agreed.skip_drafts
                else:
                    assert coordinator.decision.budgets is None
                    assert manager.get_dp_sync_state() is None
            assert manager.get_num_tokens(tokens, drafts, has_prefill=True) == 4
            cu_logits, query_lens, *_ = manager.reallocate_drafts(
                req_ids, batch.idx_mapping
            )
            assert cu_logits.tolist() == [0, 2, 4]
            assert query_lens[:3].tolist() == [0, 2, 4]
            # Filling a clipped logit capacity is not a full uniform query.
            clipped = BudgetProposal(2, 2, np.array([2.0, 3.0, 4.0]))
            decision = coordinator.coordinate(
                clipped,
                manager.online_cost_model,
                manager.runtime_timing,
                profiling=False,
            )
            assert decision.budgets == (2, 2)
            assert manager.get_dp_sync_state() is None
            # The same total can also contain extra non-drafts and fewer drafts.
            extra_base = BudgetProposal(2, 6, np.array([2.0, 3.0, 4.0]))
            decision = coordinator.coordinate(
                extra_base,
                manager.online_cost_model,
                manager.runtime_timing,
                profiling=False,
            )
            assert decision.context.num_valid_tokens_across_dp == (8, 8)
            assert manager.get_dp_sync_state() is None
            idle = BudgetProposal(0, 1, np.zeros(1)) if rank >= 4 else clipped
            coordinator.coordinate(
                idle,
                manager.online_cost_model,
                manager.runtime_timing,
                profiling=False,
            )
            assert manager.get_dp_sync_state() is None
        finally:
            cleanup_dist_env_and_memory()


def _check_online_moe_group(manager, rank, dp_rank, device):
    list(manager.batches_to_profile([1, 2, 4]))
    factor = float(rank + 1)
    manager.set_cost_curves(
        [(1, factor), (2, factor)],
        [(1, 10.0), (2, 2.0), (4, 3.0)],
        online_forward_curve=[(1, 10 * factor), (2, 2 * factor), (4, 3 * factor)],
    )
    coordinator = manager.parallel_costs
    assert coordinator is not None
    assert manager.cost_tables[0][1] == 2 * dp_rank + 1
    timing, model = manager.runtime_timing, manager.online_cost_model
    assert (model is not None) == (rank == 0)
    if model is not None:
        assert model.get_costs(2, np.array([4])).tolist() == [7.5]
    proposal = BudgetProposal(
        2 if dp_rank == 0 else 1,
        2 if dp_rank == 0 else 1,
        np.array([2.0, 2.9, 3.7]) if dp_rank == 0 else np.array([1.0, 1.9, 2.8, 3.7]),
    )
    if not get_tp_group().is_first_rank:
        proposal = replace(proposal, accepted=proposal.accepted * 0.5)
    first = coordinator.coordinate(proposal, model, timing, profiling=False)
    assert first.budgets == (2, 3)
    assert first.context.num_reqs_across_dp == (2, 1)
    assert first.context.num_tokens_across_dp == (4, 4)
    if timing is not None:
        timing.start(4, 2, context=first.context)
        torch.ones(128, device=device).sin_()
        timing.forward_end()
        timing.sampler_start()
        torch.ones(128, device=device).cos_()
        timing.sampler_end()
    torch.accelerator.synchronize()
    second = coordinator.coordinate(proposal, model, timing, profiling=False)
    assert second.context.cost_version == 0
    third = coordinator.coordinate(proposal, model, timing, profiling=False)
    assert third.context.cost_version == 1
    if model is not None:
        assert model.num_samples == 1
    # A follower's profiler vote excludes the entire cooperating step.
    profiled = coordinator.coordinate(proposal, model, timing, profiling=rank == 3)
    assert profiled.budgets is None
    assert profiled.context.epoch > second.context.epoch
    idle = BudgetProposal(0, 4, np.zeros(1)) if dp_rank == 0 else proposal
    resumed = coordinator.coordinate(idle, model, timing, profiling=False)
    assert resumed.budgets == (0, 3)
    assert resumed.context.num_reqs_across_dp == (0, 1)
    mixed = BudgetProposal(
        proposal.num_reqs,
        proposal.base_tokens,
        proposal.accepted,
        eligible=dp_rank == 0,
    )
    assert coordinator.coordinate(mixed, model, timing, profiling=False).budgets is None


@multi_gpu_test(num_gpus=4)
def test_online_tp_dp_moe_group_handles_idle_and_profiler_transitions(tmp_path):
    mp.spawn(
        _online_tp_dp_worker,
        args=(2, str(tmp_path / "moe-cost-store"), True, True),
        nprocs=4,
        join=True,
    )


def _online_tp_dp_worker(
    rank: int, tp_size: int, store: str, feedback=True, moe=False
) -> None:
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
                coordinate_moe_costs=moe,
            )
            if moe:
                _check_online_moe_group(manager, rank, dp_rank, device)
                return
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
            # At acceptance 0.5, FULL's initial zero sampler cost ties at 1.25;
            # the fixed startup draft cost of 1 ties at 1.5 instead.
            tie_cost = 1.25 if feedback else 1.5
            budget_layout = layout([[first], [0.1]], [[first], [0.1]], tie_cost)
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
                model.update([RuntimeTimingSample(1 + 3 * (dp_rank + 1), 1, 3, 2)] * 10)
                assert model.num_samples == 10
            # Followers have no local calibration, yet must obey the learned budget.
            calibrated_total = manager.get_num_tokens(
                {"low": 2, "high": 2}, {"low": [0], "high": [0]}
            )
            assert calibrated_total == (2 if feedback else 3)

            # Online decode excludes eager candidates even when their old cost
            # would beat all corrected FULL candidates. Static mode still sees
            # all four candidates and admits both drafts.
            list(manager.batches_to_profile([1, 2, 3]))
            manager.set_cost_curves(
                [(1, 1.0), (2, 1.0)],
                [(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)],
            )
            if manager.online_cost_model is not None:
                manager.online_cost_model.update(
                    [RuntimeTimingSample(31.0, 1.0, 3, 2)] * 10
                )
            mixed_range_total = manager.get_num_tokens(
                {"low": 2, "high": 2}, {"low": [0], "high": [0]}
            )
            assert mixed_range_total == (2 if feedback else 4)
            prefill_total = manager.get_num_tokens(
                {"low": 2, "high": 2}, {"low": [0], "high": [0]}, has_prefill=True
            )
            assert prefill_total == 4
            list(manager.batches_to_profile([1]))
            manager.set_cost_curves(
                [(1, 1.0), (2, 1.0)],
                [(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)],
            )
            if manager.online_cost_model is not None:
                manager.online_cost_model.update(
                    [RuntimeTimingSample(11.0, 1.0, 1, 2)] * 10
                )
            overflow_total = manager.get_num_tokens(
                {"low": 2, "high": 2}, {"low": [0], "high": [0]}
            )
            assert overflow_total == 4
            layouts = [None] * 4
            dist.all_gather_object(
                layouts,
                (
                    budget_layout,
                    boundary_layout,
                    calibrated_total,
                    mixed_range_total,
                    prefill_total,
                    overflow_total,
                ),
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
    """Output-copy completion gates timing reuse without holding the main stream."""
    collector = RuntimeTimingCollector()
    flag = torch.ones((), dtype=torch.int32, device="cuda")
    source = torch.arange(4, dtype=torch.int32, device="cuda")
    output = torch.empty(4, dtype=torch.int32, pin_memory=True)
    sampler_work = torch.zeros((), device="cuda")
    # Compile before holding the sampling stream: compilation may synchronize.
    _wait_for_runtime_timing_release[(1,)](flag, num_warps=1)
    sampler_work.add_(1)
    output.copy_(source, non_blocking=True)
    torch.accelerator.synchronize()
    flag.zero_()
    initialized = torch.cuda.Event()
    initialized.record()
    sampling_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()
    release_stream = torch.cuda.Stream()
    sampling_stream.wait_event(initialized)
    copy_stream.wait_event(initialized)
    release_stream.wait_event(initialized)
    copy_done = torch.cuda.Event(enable_timing=True)
    sampler_reference = torch.cuda.Event(enable_timing=True)
    queued = torch.cuda.Event()
    returned = Event()
    polled: list[list[RuntimeTimingSample]] = []
    errors: list[Exception] = []

    def poll():
        try:
            # A sampler_end implementation must not put the copy dependency on
            # the main stream, and collect_ready must not synchronize it either.
            queued.synchronize()
            polled.append(collector.collect_ready())
        except Exception as error:
            errors.append(error)
        finally:
            returned.set()

    thread = Thread(target=poll, daemon=True)
    try:
        with torch.cuda.stream(copy_stream):
            _wait_for_runtime_timing_release[(1,)](flag, num_warps=1)
            output.copy_(source, non_blocking=True)
            copy_done.record()
        with torch.cuda.stream(sampling_stream):
            for sample in range(1, 18):
                collector.start(num_target_tokens=sample * 8, num_reqs=1)
                collector.forward_end()
                collector.sampler_start()
                if sample == 1:
                    sampler_reference.record()
                sampler_work.add_(1)
                collector.sampler_end(copy_event=copy_done)
            queued.record()
        assert not copy_done.query()
        thread.start()
        returned_before_release = returned.wait(timeout=5)
    finally:
        with torch.cuda.stream(release_stream):
            flag.fill_(1)
        release_stream.synchronize()
        copy_stream.synchronize()
        sampling_stream.synchronize()
        torch.accelerator.synchronize()
        if thread.ident is not None:
            thread.join(timeout=5)

    assert returned_before_release, "timing collection blocked the main stream or host"
    assert not errors
    assert polled == [[]]
    completed = collector.collect_ready()
    assert [sample.num_target_tokens for sample in completed] == list(range(8, 129, 8))
    assert all(isinstance(sample, RuntimeTimingSample) for sample in completed)
    assert all(sample.num_reqs == 1 for sample in completed)
    assert all(
        sample.forward_ms >= 0 and sample.sampler_ms >= 0 for sample in completed
    )
    assert completed[0].sampler_ms >= sampler_reference.elapsed_time(copy_done)
    assert output.tolist() == [0, 1, 2, 3]
    assert collector.collect_ready() == []

    collector.start(num_target_tokens=8, num_reqs=1)
    collector.cancel()
    collector.forward_end()
    collector.sampler_start()
    collector.sampler_end()
    torch.accelerator.synchronize()
    assert collector.collect_ready() == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
def test_runtime_timing_completion_survives_a_reused_copy_event():
    """A later copy must not turn a completed sample back into pending work."""
    collector = RuntimeTimingCollector()
    flag = torch.ones((), dtype=torch.int32, device="cuda")
    _wait_for_runtime_timing_release[(1,)](flag, num_warps=1)
    copy_done = torch.cuda.Event()
    collector.start(num_target_tokens=8, num_reqs=1)
    collector.forward_end()
    collector.sampler_start()
    copy_done.record()
    copy_marker = collector.record_copy_end(torch.cuda.current_stream())
    assert copy_marker is not None
    collector.sampler_end(copy_event=copy_marker)
    torch.accelerator.synchronize()

    flag.zero_()
    initialized = torch.cuda.Event()
    initialized.record()
    later_copy = torch.cuda.Stream()
    release_stream = torch.cuda.Stream()
    later_copy.wait_event(initialized)
    release_stream.wait_event(initialized)
    returned = Event()
    polled: list[list[RuntimeTimingSample]] = []
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
        with torch.cuda.stream(later_copy):
            _wait_for_runtime_timing_release[(1,)](flag, num_warps=1)
            copy_done.record()
        assert not copy_done.query()
        thread.start()
        returned_before_release = returned.wait(timeout=5)
    finally:
        with torch.cuda.stream(release_stream):
            flag.fill_(1)
        release_stream.synchronize()
        later_copy.synchronize()
        torch.accelerator.synchronize()
        if thread.ident is not None:
            thread.join(timeout=5)

    assert returned_before_release, "reused copy event blocked an old timing sample"
    assert not errors
    assert len(polled) == 1 and len(polled[0]) == 1
    assert polled[0][0].num_target_tokens == 8
    assert polled[0][0].num_reqs == 1
    assert collector.collect_ready() == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
def test_runtime_timing_invalidation_keeps_old_dependencies_and_new_samples():
    """Session changes discard old data without releasing unfinished slots."""
    collector = RuntimeTimingCollector()
    flag = torch.ones((), dtype=torch.int32, device="cuda")
    _wait_for_runtime_timing_release[(1,)](flag, num_warps=1)
    torch.accelerator.synchronize()
    flag.zero_()
    initialized = torch.cuda.Event()
    initialized.record()
    copy_stream = torch.cuda.Stream()
    release_stream = torch.cuda.Stream()
    copy_stream.wait_event(initialized)
    release_stream.wait_event(initialized)
    copy_done = torch.cuda.Event()
    returned = Event()
    polled: list[list[RuntimeTimingSample]] = []
    errors: list[Exception] = []

    def wait_for_second(end):
        try:
            for marker in end:
                marker.synchronize()
            polled.append(collector.collect_ready())
        except Exception as error:
            errors.append(error)
        finally:
            returned.set()

    thread = None
    try:
        collector.start(num_target_tokens=8, num_reqs=1)
        collector.forward_end()
        collector.sampler_start()
        with torch.cuda.stream(copy_stream):
            _wait_for_runtime_timing_release[(1,)](flag, num_warps=1)
            copy_done.record()
        copy_marker = collector.record_copy_end(copy_stream)
        assert copy_marker is not None
        first_end = collector.sampler_end(copy_event=copy_marker)
        collector.invalidate()
        collector.start(num_target_tokens=16, num_reqs=1)
        collector.forward_end()
        collector.sampler_start()
        second_end = collector.sampler_end()
        assert first_end is not None and second_end is not None
        thread = Thread(target=wait_for_second, args=(second_end,), daemon=True)
        thread.start()
        returned_before_release = returned.wait(timeout=5)
        first_still_pending = any(not marker.query() for marker in first_end)
    finally:
        with torch.cuda.stream(release_stream):
            flag.fill_(1)
        release_stream.synchronize()
        copy_stream.synchronize()
        torch.accelerator.synchronize()
        if thread is not None:
            thread.join(timeout=5)

    assert returned_before_release, "second step waited for the first step's copy"
    assert first_still_pending, "first step completed without its copy dependency"
    assert not errors
    assert polled == [[]], "invalidation released the unfinished old queue entry"
    assert [sample.num_target_tokens for sample in collector.collect_ready()] == [16]
