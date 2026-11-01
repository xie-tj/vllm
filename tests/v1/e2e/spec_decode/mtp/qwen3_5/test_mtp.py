# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from tests.utils import RemoteOpenAIServer, multi_gpu_test, single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig, CUDAGraphMode, KernelConfig, ModelConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm.platforms import current_platform

from ...utils import (
    assert_request_outputs_match,
    compute_acceptance_len,
    get_spec_decode_metric_value,
    get_test_prompts,
)
from .._correctness import check_mtp_correctness

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class VocabSamplingTestExtension:
    model_runner: "GPUModelRunner"

    def prepare_vocab_equivalence(self):
        import torch

        from vllm.distributed import get_ep_group
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
        from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
            compute_acceptance_partials,
        )
        from vllm.v1.worker.gpu.spec_decode.vocab_stats import gather_draft_statistics

        runner = self.model_runner
        assert runner.adaptive_verification is not None
        assert runner.speculator is not None
        config = runner.vllm_config
        assert config.model_config is not None
        width = config.model_config.hf_text_config.hidden_size
        hidden = (
            (torch.arange(2 * width, device=runner.device).reshape(2, width) % 37 - 18)
            / 32
        ).to(config.model_config.dtype)
        mapping = torch.arange(2, device=runner.device)
        temperature = torch.zeros(2, device=runner.device)
        seeds = torch.zeros(2, dtype=torch.int64, device=runner.device)
        target_logits = runner.model.compute_logits(hidden)
        assert target_logits is not None
        target_ids = runner.model.get_top_tokens(hidden)
        assert torch.equal(target_ids, target_logits.argmax(-1))
        assert torch.equal(
            target_ids,
            gumbel_sample(
                target_logits,
                mapping,
                temperature,
                seeds,
                mapping,
                apply_temperature=False,
                is_drafting=False,
            ),
        )
        draft_logits = runner.speculator.model.compute_logits(hidden)
        assert draft_logits is not None
        ids, maxima, sums = gather_draft_statistics(
            runner.speculator.model.compute_logits_local(hidden),
            mapping,
            temperature,
            draft_logits.shape[-1],
        )
        ref_maxima, ref_sums = compute_acceptance_partials(
            draft_logits, mapping, temperature
        )
        assert torch.equal(ids, draft_logits.argmax(-1))
        torch.testing.assert_close(maxima, ref_maxima, rtol=0, atol=0)
        torch.testing.assert_close(sums, ref_sums, rtol=0, atol=0)
        # Keep identical flat startup prices across servers. Verification stays
        # adaptive; DP routing is pinned below. Never use these prices for perf.
        runner.adaptive_verification.set_cost_curves(
            [(1, 0.0), (128, 0.0)],
            [(1, 1.0), (512, 1.0)],
        )
        assert config.speculative_config is not None
        comm = get_ep_group().device_communicator
        assert comm is not None
        return {
            "vocab_statistics": runner.speculator.use_vocab_statistics,
            "dispatch_packing": getattr(comm.all2all_manager, "pack_dispatch", False),
            "vocab_sampling_apis": (
                hasattr(runner.model, "get_top_tokens")
                and hasattr(runner.speculator.model, "compute_logits_local")
            ),
            "local_argmax": config.speculative_config.use_local_argmax_reduction,
            "batch_sharding": runner.batch_sharder is not None,
            "prefix_caching": config.cache_config.enable_prefix_caching,
            "chunked_prefill": config.scheduler_config.enable_chunked_prefill,
            "jit_warmup": config.kernel_config.enable_jit_warmup,
            "moe_backend": config.kernel_config.moe_backend,
        }


@pytest.mark.distributed(num_gpus=8)
@pytest.mark.asyncio
async def test_vocab_parallel_sampling_matches_native_tp4_dp2_ep(monkeypatch):
    """Real greedy outputs and constrained/logprob fallbacks retain native AV."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model = os.environ.get("VLLM_TEST_MOE_MTP_MODEL", "Qwen/Qwen3.5-35B-A3B-FP8")
    results, configurations = [], []
    for optimized in (False, True):
        args = [
            "--served-model-name",
            "vocab-av",
            "--tensor-parallel-size",
            "4",
            "--data-parallel-size",
            "2",
            "--data-parallel-size-local",
            "2",
            "--enable-expert-parallel",
            "--distributed-executor-backend",
            "mp",
            "--language-model-only",
            "--max-model-len",
            "4096",
            "--max-num-batched-tokens",
            "512",
            "--max-num-seqs",
            "128",
            "--gpu-memory-utilization",
            "0.80",
            "--kernel-config",
            json.dumps({"enable_moe_dispatch_packing": optimized}),
            "--worker-extension-cls",
            f"{__name__}.VocabSamplingTestExtension",
            "--speculative-config",
            json.dumps(
                {
                    "method": "mtp",
                    "num_speculative_tokens": 3,
                    "enable_adaptive_verification": True,
                    "enable_vocab_parallel_sampling": optimized,
                }
            ),
        ]
        with RemoteOpenAIServer(
            model,
            args,
            env_dict={
                "VLLM_USE_V2_MODEL_RUNNER": "1",
                "VLLM_SERVER_DEV_MODE": "1",
                "PYTHONPATH": os.pathsep.join(
                    [
                        str(Path(__file__).resolve().parents[6]),
                        os.environ.get("PYTHONPATH", ""),
                    ]
                ),
            },
            max_wait_seconds=900,
        ) as server:
            async with (
                server.get_async_client(
                    default_headers={"X-data-parallel-rank": "0"}
                ) as client,
                httpx.AsyncClient() as rpc,
            ):
                response = await rpc.post(
                    server.url_for("collective_rpc"),
                    json={"method": "prepare_vocab_equivalence"},
                    timeout=120,
                )
                response.raise_for_status()
                states = response.json()["results"]
                states = [
                    s
                    for row in states
                    for s in (row if isinstance(row, list) else [row])
                ]
                assert all(s.pop("vocab_statistics") == optimized for s in states)
                assert all(s.pop("dispatch_packing") == optimized for s in states)
                assert all(s.pop("vocab_sampling_apis") for s in states)
                configurations.append(states)
                constrained = await client.completions.create(
                    model="vocab-av",
                    prompt=["Continue."] * 16,
                    temperature=0,
                    max_tokens=16,
                    extra_body={
                        "ignore_eos": True,
                        "allowed_token_ids": [42],
                        "return_token_ids": True,
                    },
                )
                assert len(constrained.choices) == 16
                assert all(
                    choice.token_ids == [42] * 16 for choice in constrained.choices
                )
                tokens = []
                for prompt in (
                    "The number after one is",
                    "Write the weekdays in order.",
                ):
                    output = await client.completions.create(
                        model="vocab-av",
                        prompt=prompt,
                        temperature=0,
                        max_tokens=32,
                        extra_body={"ignore_eos": True, "return_token_ids": True},
                    )
                    assert len(output.choices[0].token_ids) == 32
                    reference = await client.completions.create(
                        model="vocab-av",
                        prompt=prompt,
                        temperature=0,
                        max_tokens=32,
                        logprobs=0,
                        extra_body={"ignore_eos": True, "return_token_ids": True},
                    )
                    assert output.choices[0].token_ids == reference.choices[0].token_ids
                    tokens.append(output.choices[0].token_ids)
                results.append(tokens)
                scored = await client.completions.create(
                    model="vocab-av",
                    prompt="Continue.",
                    temperature=0,
                    max_tokens=4,
                    logprobs=0,
                    extra_body={"ignore_eos": True},
                )
                assert len(scored.choices[0].logprobs.token_logprobs) == 4
    assert configurations[0] == configurations[1]
    assert results[0] == results[1]


@pytest.mark.parametrize(
    ("architecture", "method"),
    [
        ("Qwen3_5MoeForCausalLM", "get_top_tokens"),
        ("Qwen3_5MoeForConditionalGeneration", "get_top_tokens"),
        ("Qwen3_5MoeMTP", "compute_logits_local"),
    ],
)
def test_vocab_sampling_registered_model_apis(architecture, method):
    """The platform-selected class must expose the optimization's model APIs."""
    model = os.environ.get("VLLM_TEST_MOE_MTP_MODEL", "Qwen/Qwen3.5-35B-A3B-FP8")
    config = ModelConfig(model=model, max_model_len=512)
    cls, _ = ModelRegistry.resolve_model_cls(architecture, config)
    assert callable(getattr(cls, method, None)), (
        f"{cls.__module__}.{cls.__name__}.{method}"
    )


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


@pytest.mark.parametrize("dynamic_steps", [False, True], ids=["fixed", "dynamic"])
@multi_gpu_test(num_gpus=2)
def test_online_cost_feedback_moe_ep_without_dp_uses_tp_driver(
    monkeypatch, vllm_runner, enable_pickle, dynamic_steps
):
    """EP with one batch retains native settings and learns only on the TP driver."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model = os.environ.get("VLLM_TEST_MOE_MTP_MODEL", "Qwen/Qwen3.5-35B-A3B-FP8")
    with vllm_runner(
        model,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
        max_model_len=512,
        max_num_batched_tokens=128,
        max_num_seqs=2,
        block_size=None,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
        kernel_config=KernelConfig(),
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
            "enable_adaptive_verification": True,
            "enable_online_cost_feedback": True,
            **(
                {"num_speculative_tokens_per_batch_size": [(1, 1, 3), (2, 2, 1)]}
                if dynamic_steps
                else {}
            ),
        },
    ) as runner:
        params = SamplingParams(
            temperature=0,
            max_tokens=128,
            ignore_eos=True,
            allowed_token_ids=[42],
        )
        outputs = runner.llm.generate(
            ["Continue the following: hello"],
            params,
        )
        assert list(outputs[0].outputs[0].token_ids) == [42] * 128
        if dynamic_steps:
            outputs = runner.llm.generate(
                ["Continue the following: first", "Continue the following: second"],
                params,
            )
            assert all(list(o.outputs[0].token_ids) == [42] * 128 for o in outputs)

        def state(worker):
            from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

            model_runner = worker.model_runner
            manager = model_runner.adaptive_verification
            cost_model = manager.online_cost_model
            config = model_runner.vllm_config
            return {
                "coordinator_bypassed": manager.parallel_costs is None,
                "samples": None if cost_model is None else cost_model.num_samples,
                "random_routing_active": any(
                    m.profile_random_routing is not None
                    for m in model_runner.model.modules()
                    if isinstance(m, MoERunner)
                ),
                "schedule": (
                    config.speculative_config.num_speculative_tokens_per_batch_size
                ),
                "dp_size": config.parallel_config.data_parallel_size,
                "prefix_caching": config.cache_config.enable_prefix_caching,
                "chunked_prefill": config.scheduler_config.enable_chunked_prefill,
                "moe_backend": config.kernel_config.moe_backend,
                "jit_warmup": config.kernel_config.enable_jit_warmup,
            }

        states = runner.llm.collective_rpc(state)
        print("DP1_STATE " + json.dumps(states))
        assert all(
            s["coordinator_bypassed"] and not s["random_routing_active"] for s in states
        )
        assert all(s["dp_size"] == 1 for s in states)
        expected_schedule = [(1, 1, 3), (2, 2, 1)] if dynamic_steps else None
        assert all(s["schedule"] == expected_schedule for s in states)
        assert states[0]["samples"] >= 10
        assert states[1]["samples"] is None


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
