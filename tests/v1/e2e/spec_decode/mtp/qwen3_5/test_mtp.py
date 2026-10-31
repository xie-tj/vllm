# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import os

import httpx
import pytest

from tests.utils import RemoteOpenAIServer, multi_gpu_test, single_gpu_only
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
    """Full sampler timing and adaptive budgets preserve constrained outputs."""
    _check_online_mtp_generation(monkeypatch, vllm_runner, online_cost_feedback)


@multi_gpu_test(num_gpus=2)
def test_online_cost_feedback_tensor_parallel_generation(
    monkeypatch: pytest.MonkeyPatch, vllm_runner, enable_pickle
):
    """TP followers obey the leader's calibrated verification decisions."""
    _check_online_mtp_generation(monkeypatch, vllm_runner, True, tp_size=2)


@multi_gpu_test(num_gpus=1)
def test_online_cost_feedback_native_backend_generation(
    monkeypatch, vllm_runner, enable_pickle
):
    """The native backend selector finds a compatible variable-query backend."""
    _check_online_mtp_generation(monkeypatch, vllm_runner, True, native_settings=True)


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.asyncio
@pytest.mark.parametrize("online_cost_feedback", [False, True], ids=["off", "on"])
@pytest.mark.parametrize("expert_parallel", [False, True], ids=["noep", "ep"])
async def test_moe_mtp_dp_survives_idle_transitions(
    monkeypatch, tmp_path, online_cost_feedback, expert_parallel
):
    """An idle replica preserves the target's batch shape during MTP prefill."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model = os.environ.get("VLLM_TEST_MOE_MTP_MODEL", "Qwen/Qwen3.5-35B-A3B-FP8")
    args = [
        "--served-model-name",
        "moe-mtp-dp",
        "--data-parallel-size",
        "2",
        "--data-parallel-size-local",
        "2",
        "--distributed-executor-backend",
        "mp",
        "--disable-custom-all-reduce",
        "--language-model-only",
        "--attention-backend",
        "TRITON_ATTN",
        "--gdn-prefill-backend",
        "triton",
        "--moe-backend",
        "triton",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "4096",
        "--max-num-batched-tokens",
        "256",
        "--max-num-seqs",
        "32",
        "--gpu-memory-utilization",
        "0.80",
        "--compilation-config",
        '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
        "--speculative-config",
        json.dumps(
            {
                "method": "mtp",
                "num_speculative_tokens": 3,
                "enable_adaptive_verification": True,
                "use_local_argmax_reduction": False,
                "enable_online_cost_feedback": online_cost_feedback,
            }
        ),
    ]
    if expert_parallel:
        args += [
            "--enable-expert-parallel",
            "--all2all-backend",
            "allgather_reducescatter",
        ]
    if online_cost_feedback:
        args += [
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str(tmp_path / "profiles"),
                    "torch_profiler_with_stack": False,
                    "ignore_frontend": True,
                }
            ),
        ]
    with RemoteOpenAIServer(
        model,
        args,
        env_dict={
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_USE_DEEP_GEMM": "0",
            "VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN": "512",
            "VLLM_SERVER_DEV_MODE": "1",
        },
        max_wait_seconds=900,
    ) as server:
        async with server.get_async_client() as client, httpx.AsyncClient() as health:

            async def generate(index, tokens=64, prompt=None):
                response = await client.completions.create(
                    model="moe-mtp-dp",
                    prompt=prompt or f"Request {index}: continue.",
                    temperature=0,
                    max_tokens=tokens,
                    extra_body={
                        "ignore_eos": True,
                        "allowed_token_ids": [42],
                        "return_token_ids": True,
                    },
                )
                assert response.choices[0].token_ids == [42] * tokens

            for count in (32, 1, 8):
                await asyncio.gather(*(generate(i) for i in range(count)))
                response = await health.get(server.url_for("health"))
                assert response.status_code == 200, response.text
            await asyncio.gather(
                generate(0, 32, "Continue the following: " + "hello " * 300),
                generate(1, 128),
            )
            stream = await client.completions.create(
                model="moe-mtp-dp",
                prompt="Continue.",
                max_tokens=2048,
                temperature=0,
                stream=True,
                extra_body={"ignore_eos": True},
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].text:
                    break
            await stream.close()
            await generate(2, 32)

            async def cost_stats():
                response = await health.post(
                    server.url_for("collective_rpc"),
                    json={"method": "get_online_cost_feedback_stats"},
                    timeout=60,
                )
                response.raise_for_status()
                values = response.json()["results"]
                return [
                    v
                    for row in values
                    for v in (row if isinstance(row, list) else [row])
                ]

            before = await cost_stats()
            if online_cost_feedback:
                drivers = [s for s in before if s.get("driver")]
                assert len(drivers) == 1
                assert drivers[0]["num_samples"] >= 10
                assert all(
                    s["num_samples"] is None for s in before if not s.get("driver")
                )
                response = await health.post(
                    server.url_for("start_profile"), timeout=180
                )
                response.raise_for_status()
                await generate(3, 32)
                response = await health.post(
                    server.url_for("stop_profile"), timeout=180
                )
                response.raise_for_status()
                await generate(4, 64)
                after = await cost_stats()
                driver = next(s for s in after if s.get("driver"))
                assert driver["epoch"] >= drivers[0]["epoch"] + 2
                for observation in driver["observations"]:
                    context = observation["context"]
                    assert len(context["num_reqs_across_dp"]) == 2
                    assert context["num_tokens_across_dp"][0] > 0
                    assert context["epoch"] != drivers[0]["epoch"] + 1
            else:
                after = before
                assert all(s == {"num_samples": None} for s in before)
            (tmp_path / "online-cost-stats.json").write_text(
                json.dumps(
                    {
                        "feedback": online_cost_feedback,
                        "ep": expert_parallel,
                        "before_profile": before,
                        "after_profile": after,
                    },
                    indent=2,
                )
            )
            response = await health.get(server.url_for("health"))
            assert response.status_code == 200, response.text


def _check_online_mtp_generation(
    monkeypatch, vllm_runner, online_cost_feedback, tp_size=1, native_settings=False
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
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        attention_backend=None if native_settings else "TRITON_ATTN",
        compilation_config=CompilationConfig()
        if native_settings
        else CompilationConfig(
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
            # Real decode steps feed forward and full-sampler observations.
            assert observed_samples[0] >= 10
            assert observed_samples[1:] == [None] * (tp_size - 1)
        else:
            assert observed_samples == [None] * tp_size
        if native_settings:

            def dummy_geometry(worker):
                runner = worker.model_runner
                with runner.step_timing.collect() as samples:
                    runner._dummy_run(4, uniform_decode=True)
                return [(s.num_target_tokens, s.num_reqs) for s in samples]

            assert runner.llm.collective_rpc(dummy_geometry) == [[(4, 1)]]


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
