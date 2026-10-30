# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import multi_gpu_test, single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.platforms import current_platform

from ...utils import (
    assert_request_outputs_match,
    compute_acceptance_len,
    get_spec_decode_metric_value,
    get_test_prompts,
)
from .._correctness import check_mtp_correctness


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        (
            ("mtp", "Qwen/Qwen3.5-0.8B-Base", 1, None),
            False,
            0.20,
        ),  # hybrid + MTP, ref: ~34%-35%
    ],
    ids=["qwen3_5-hybrid"],
)
@single_gpu_only
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int, str | None],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    vllm_runner,
):
    check_mtp_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        vllm_runner,
    )


@pytest.mark.parametrize("online_cost_feedback", [False, True])
@single_gpu_only
def test_online_cost_feedback_preserves_constrained_mtp_generation(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner,
    enable_pickle,
    online_cost_feedback: bool,
):
    """Runtime timing and adaptive budgets must preserve constrained outputs."""
    _check_online_mtp_generation(monkeypatch, vllm_runner, online_cost_feedback)


@multi_gpu_test(num_gpus=2)
def test_online_cost_feedback_tensor_parallel_generation(
    monkeypatch: pytest.MonkeyPatch, vllm_runner, enable_pickle
):
    """TP followers obey the leader's calibrated verification decisions."""
    _check_online_mtp_generation(monkeypatch, vllm_runner, True, tp_size=2)


def _check_online_mtp_generation(
    monkeypatch, vllm_runner, online_cost_feedback, tp_size=1
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", "64")
    params = SamplingParams(
        temperature=0.0,
        max_tokens=384,
        ignore_eos=True,
        allowed_token_ids=[42],
    )
    with vllm_runner(
        "Qwen/Qwen3.5-0.8B-Base",
        tensor_parallel_size=tp_size,
        max_model_len=512,
        max_num_batched_tokens=128,
        max_num_seqs=2,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        attention_backend="TRITON_ATTN",
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[1, 4, 8],
        ),
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
            "enable_adaptive_verification": True,
            "enable_online_cost_feedback": online_cost_feedback,
            "use_local_argmax_reduction": False,
        },
        disable_log_stats=False,
    ) as runner:
        outputs = runner.llm.generate(["The capital of France is"], params)
        assert len(outputs) == 1
        assert list(outputs[0].outputs[0].token_ids) == [42] * 384
        assert (
            get_spec_decode_metric_value(
                runner.llm.get_metrics(), "vllm:spec_decode_num_drafts"
            )
            > 0
        )

        def num_cost_samples(worker):
            model = worker.model_runner.adaptive_verification.online_cost_model
            return None if model is None else model.num_samples

        observed_samples = runner.llm.collective_rpc(num_cost_samples)
        if online_cost_feedback:
            assert len(observed_samples) == tp_size
            assert observed_samples[0] >= 10
            assert observed_samples[1:] == [None] * (tp_size - 1)
        else:
            assert observed_samples == [None] * tp_size


@single_gpu_only
def test_mtp_draft_lm_head_quantization(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    vllm_runner,
):
    """A quantized draft copy of the shared lm_head only changes drafting:
    greedy outputs match stock MTP and acceptance stays within 10%."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    prompts = get_test_prompts(mm_enabled=False)
    results = {}
    # NVFP4 needs a CUDA W4A16 kernel.
    quants = (None, "fp8", "nvfp4") if current_platform.is_cuda() else (None, "fp8")
    for quant in quants:
        with vllm_runner(
            "Qwen/Qwen3.5-0.8B-Base",
            block_size=None,
            enable_chunked_prefill=None,
            max_model_len=2048,
            limit_mm_per_prompt={"image": 0, "video": 0},
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 3,
                "draft_lm_head_quantization": quant,
            },
            disable_log_stats=False,
        ) as runner:
            outputs = runner.llm.chat(prompts, sampling_config)
            results[quant] = (outputs, compute_acceptance_len(runner.llm.get_metrics()))

    ref_outputs, ref_al = results.pop(None)
    for quant, (outputs, al) in results.items():
        assert_request_outputs_match(
            ref_outputs,
            outputs,
            # Not bit-exact: a different acceptance length changes the target's
            # verify batch shapes, which can flip a near-tied greedy token.
            required_matches=int(0.8 * len(ref_outputs)) + 1,
            context=f"draft_lm_head_quantization={quant}",
        )
        assert al >= 0.9 * ref_al, (
            f"{quant}: acceptance length {al:.3f} vs {ref_al:.3f}"
        )


@single_gpu_only
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="nvfp4 needs a CUDA W4A16 kernel"
)
def test_mtp_draft_lm_head_quantization_survives_sleep_and_reload(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    vllm_runner,
):
    """The quantized draft head is derived from the target head: it must come
    back intact after a level 1 sleep and be re-derived after a target weight
    reload, so outputs and acceptance match the run before."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    prompts = get_test_prompts(mm_enabled=False)
    with vllm_runner(
        "Qwen/Qwen3.5-0.8B-Base",
        block_size=None,
        enable_chunked_prefill=None,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
            "draft_lm_head_quantization": "nvfp4",
        },
        enable_sleep_mode=True,
        disable_log_stats=False,
    ) as runner:
        llm = runner.llm

        def run():
            before = llm.get_metrics()
            outputs = llm.chat(prompts, sampling_config)
            return outputs, compute_acceptance_len(llm.get_metrics(), before)

        ref_outputs, ref_al = run()

        llm.sleep(level=1)
        llm.wake_up()
        outputs, al = run()
        assert_request_outputs_match(
            ref_outputs,
            outputs,
            required_matches=int(0.8 * len(ref_outputs)) + 1,
            context="after level 1 sleep",
        )
        assert al == pytest.approx(ref_al, rel=0.03)

        llm.collective_rpc("reload_weights")
        outputs, al = run()
        assert_request_outputs_match(
            ref_outputs,
            outputs,
            required_matches=int(0.8 * len(ref_outputs)) + 1,
            context="after weight reload",
        )
        assert al == pytest.approx(ref_al, rel=0.03)
