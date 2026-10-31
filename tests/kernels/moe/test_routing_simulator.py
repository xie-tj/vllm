#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test script for the token-to-expert routing simulator.

This script demonstrates how to use the routing simulator to test
different routing strategies and analyze their performance, including
integration tests with FusedMoEFactory layer.
"""

import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
    DistributionBasedRouting,
    RoutingSimulator,
    RoutingSimulatorRouter,
)
from vllm.platforms import current_platform


@pytest.fixture
def device():
    """Fixture to provide the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("num_tokens", [1, 16, 256])
@pytest.mark.parametrize("hidden_size", [64, 1024])
@pytest.mark.parametrize("num_experts", [16, 128])
@pytest.mark.parametrize("top_k", [1, 4])
def test_basic_functionality(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    device,
):
    """Test basic functionality of the routing simulator."""
    # Test each routing strategy
    strategies = RoutingSimulator.get_available_strategies()

    hidden_states = torch.randn(num_tokens, hidden_size, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    for strategy in strategies:
        # Simulate routing
        topk_weights, topk_ids = RoutingSimulator.simulate_routing(
            hidden_states=hidden_states,
            router_logits=router_logits,
            strategy_name=strategy,
            top_k=top_k,
        )

        # Check output shapes
        assert topk_weights.shape == (
            num_tokens,
            top_k,
        ), f"Wrong weights shape for {strategy}"
        assert topk_ids.shape == (
            num_tokens,
            top_k,
        ), f"Wrong ids shape for {strategy}"

        # Check that expert IDs are valid
        assert topk_ids.min() >= 0, f"Invalid expert ID (negative) for {strategy}"
        assert topk_ids.max() < num_experts, (
            f"Invalid expert ID (too large) for {strategy}"
        )


def test_routing_strategy_integration(monkeypatch, device):
    """Test that the routing strategy environment variable works with
    FusedMoEFactory."""
    pytest.importorskip("vllm.model_executor.layers.fused_moe.layer")

    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory

    # Test parameters
    num_tokens = 32
    hidden_size = 16
    num_experts = 4
    top_k = 2

    # Create test data
    hidden_states = torch.randn(num_tokens, hidden_size, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    # Test different routing strategies
    strategies = RoutingSimulator.get_available_strategies()

    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        for strategy in strategies:
            # Set environment variable
            env_name = "VLLM_MOE_ROUTING_SIMULATION_STRATEGY"
            monkeypatch.setenv(env_name, strategy)

            # Temporarily override the envs lookup so the router factory
            # reads the monkeypatched value instead of the module-load-time
            # default. Use monkeypatch.setitem so the original lambda is
            # restored automatically at teardown.
            monkeypatch.setitem(
                envs.environment_variables,
                env_name,
                lambda s=strategy: s,
            )

            fused_moe = FusedMoEFactory(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=0,
                use_grouped_topk=False,
                renormalize=True,
                prefix=strategy,
            )

            assert (
                fused_moe.moe_config.routing_method
                == RoutingMethodType.RenormalizeNaive
            )
            assert fused_moe.moe_config.require_decomposed_backend == (
                not RoutingSimulator.get_strategy(strategy).supports_monolithic_logits
            )

            # Test the select_experts method
            topk_weights, topk_ids = fused_moe.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

            # Verify output shapes
            assert topk_weights.shape == (num_tokens, top_k), (
                f"Wrong weights shape for {strategy}"
            )
            assert topk_ids.shape == (num_tokens, top_k), (
                f"Wrong ids shape for {strategy}"
            )

            # Verify expert IDs are valid
            assert topk_ids.min() >= 0, f"Invalid expert ID (negative) for {strategy}"
            assert topk_ids.max() < num_experts, (
                f"Invalid expert ID (too large) for {strategy}"
            )


def test_simulated_bias_is_nonpersistent_buffer(monkeypatch, dist_init):
    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory

    monkeypatch.setitem(
        envs.environment_variables,
        "VLLM_MOE_ROUTING_SIMULATION_STRATEGY",
        lambda: "uniform_random",
    )
    bias = torch.nn.Parameter(torch.ones(4))
    with set_current_vllm_config(VllmConfig()):
        moe = FusedMoEFactory(
            num_experts=4,
            top_k=2,
            hidden_size=128,
            intermediate_size=128,
            renormalize=True,
            scoring_func="sigmoid",
            e_score_correction_bias=bias,
            prefix="bias_registration",
        )

    experts = moe.routed_experts
    name = "e_score_correction_bias"
    assert dict(experts.named_buffers())[name] is experts.e_score_correction_bias
    assert name not in experts.state_dict()
    assert torch.count_nonzero(experts.e_score_correction_bias) == 0
    assert torch.all(bias == 1)


def test_distribution_based_routing_with_custom_strategy():
    """Test registering and using DistributionBasedRouting with custom
    parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Register custom distribution-based strategy
    custom_strategy = DistributionBasedRouting(distribution="normal", mean=2.0, std=0.5)
    RoutingSimulator.register_strategy("custom_normal", custom_strategy)

    # Test data
    num_tokens = 60
    hidden_size = 48
    num_experts = 6
    top_k = 3

    hidden_states = torch.randn(num_tokens, hidden_size, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    # Use the custom strategy
    topk_weights, topk_ids = RoutingSimulator.simulate_routing(
        hidden_states=hidden_states,
        router_logits=router_logits,
        strategy_name="custom_normal",
        top_k=top_k,
    )

    # Check output shapes
    assert topk_weights.shape == (num_tokens, top_k)
    assert topk_ids.shape == (num_tokens, top_k)

    # Check that expert IDs are valid
    assert topk_ids.min() >= 0
    assert topk_ids.max() < num_experts


def test_instance_compatibility():
    """Test that static methods work correctly."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test static method directly
    hidden_states = torch.randn(10, 8, device=device)
    router_logits = torch.randn(10, 4, device=device)

    topk_weights, topk_ids = RoutingSimulator.simulate_routing(
        hidden_states=hidden_states,
        router_logits=router_logits,
        strategy_name="uniform_random",
        top_k=2,
    )

    assert topk_weights.shape == (10, 2)
    assert topk_ids.shape == (10, 2)


@pytest.mark.parametrize(
    "groups,has_bias,expected_method",
    [
        (4, True, RoutingMethodType.DeepSeekV3),
        (None, True, RoutingMethodType.MiniMax2),
        (None, False, RoutingMethodType.Default),
    ],
)
def test_simulation_preserves_model_routing_method(
    monkeypatch, groups, has_bias, expected_method
):
    """Biased and non-renormalized models keep their kernel routing semantics."""
    import vllm.envs as envs
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setitem(
        envs.environment_variables,
        "VLLM_MOE_ROUTING_SIMULATION_STRATEGY",
        lambda: "uniform_random",
    )
    bias = torch.ones(32) if has_bias else None
    router = create_fused_moe_router(
        top_k=2,
        global_num_experts=32,
        use_grouped_topk=groups is not None,
        num_expert_group=groups,
        topk_group=1 if groups is not None else None,
        scoring_func="sigmoid" if has_bias else "softmax",
        renormalize=has_bias,
        e_score_correction_bias=bias,
    )
    assert router.routing_method_type == RoutingMethodType.Simulated
    assert router.kernel_routing_method_type == expected_method


def test_runner_prepares_monolithic_logits(monkeypatch, device):
    """The shared runner must replace planted logits before calling any backend."""
    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    monkeypatch.setitem(
        envs.environment_variables,
        "VLLM_MOE_ROUTING_SIMULATION_STRATEGY",
        lambda: "uniform_random",
    )
    router = RoutingSimulatorRouter(2, 32, RoutingMethodType.RenormalizeNaive)
    experts = SimpleNamespace(
        quant_method=SimpleNamespace(is_monolithic=True), forward_monolithic=Mock()
    )
    runner = SimpleNamespace(
        router=router,
        routed_experts=experts,
        _maybe_apply_shared_experts=Mock(),
        _shared_experts=None,
    )
    logits = torch.full((16, 32), 7.0, device=device)
    hidden = torch.zeros((16, 64), device=device)
    MoERunner._apply_quant_method(runner, hidden, logits, None)
    passed = experts.forward_monolithic.call_args.kwargs["router_logits"]
    assert passed.shape == logits.shape
    assert passed.dtype == logits.dtype
    assert passed.device == logits.device
    assert torch.all((passed >= -1) & (passed <= 0))
    assert torch.all(logits == 7)


def _deepinfer_routing_reference(num_tokens, num_experts, top_k, rank):
    """Use the independent Torch Philox implementation and sequential unranking."""
    from vllm.v1.watermarking.prfs.philox import _philox4x32_10

    lanes = torch.arange(top_k, dtype=torch.int64).unsqueeze(0)
    subsequences = torch.arange(num_tokens, dtype=torch.int64).unsqueeze(1) * 32 + lanes
    zeros = torch.zeros_like(subsequences)
    words = _philox4x32_10(
        (zeros, zeros, subsequences & 0xFFFFFFFF, subsequences >> 32),
        (
            torch.full_like(subsequences, rank & 0xFFFFFFFF),
            torch.full_like(subsequences, rank >> 32),
        ),
    )
    weights = words[0].to(torch.float32) * 2**-32 + 2**-33
    ids = []
    for token in range(num_tokens):
        selected: list[int] = []
        for lane in range(top_k):
            candidate = int(words[1][token, lane]) % (num_experts - lane)
            # Draw an index in the remaining sorted population. This avoids
            # copying the production implementation's parallel minimum loop.
            for previous in sorted(selected):
                if candidate >= previous:
                    candidate += 1
            selected.append(candidate)
        ids.append(selected)
    return weights, torch.tensor(ids, dtype=torch.int64).reshape(num_tokens, top_k)


@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("top_k", [1, 8, 10])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_deepinfer_profile_routing_matches_curand_protocol(rank, top_k, indices_dtype):
    """Physical experts, rank seeds and first two Philox words define routing."""
    from vllm.model_executor.layers.fused_moe.router.deepinfer_profile import (
        make_deepinfer_profile_routing,
    )

    # Ten physical experts can include replicas beyond eight logical experts.
    num_tokens, num_experts = 17, 10
    before = torch.get_rng_state()
    weights, ids = make_deepinfer_profile_routing(
        num_tokens, num_experts, top_k, rank, torch.device("cpu"), indices_dtype
    )
    expected_weights, expected_ids = _deepinfer_routing_reference(
        num_tokens, num_experts, top_k, rank
    )
    assert weights.dtype == torch.float32
    assert ids.dtype == indices_dtype
    assert torch.equal(weights, expected_weights)
    assert torch.equal(ids.to(torch.int64), expected_ids)
    assert torch.all((weights > 0) & (weights <= 1))
    assert torch.all((ids >= 0) & (ids < num_experts))
    sorted_ids = ids.sort(dim=-1).values
    assert torch.all(sorted_ids[:, 1:] != sorted_ids[:, :-1])
    if top_k == num_experts:
        assert torch.equal(
            sorted_ids.to(torch.int64), torch.arange(num_experts).expand(num_tokens, -1)
        )
    repeated = make_deepinfer_profile_routing(
        num_tokens, num_experts, top_k, rank, torch.device("cpu"), indices_dtype
    )
    prefix = make_deepinfer_profile_routing(
        5, num_experts, top_k, rank, torch.device("cpu"), indices_dtype
    )
    assert torch.equal(weights, repeated[0]) and torch.equal(ids, repeated[1])
    assert torch.equal(weights[:5], prefix[0]) and torch.equal(ids[:5], prefix[1])
    assert torch.equal(torch.get_rng_state(), before)


def test_deepinfer_profile_routing_first_philox_vector():
    """Lock seed/subsequence/word ordering to the published zero-counter vector."""
    from vllm.model_executor.layers.fused_moe.router.deepinfer_profile import (
        make_deepinfer_profile_routing,
    )
    from vllm.v1.watermarking.prfs.philox import _philox4x32_10

    zero = torch.tensor(0, dtype=torch.int64)
    words = _philox4x32_10((zero, zero, zero, zero), (zero, zero))
    assert tuple(int(word) for word in words) == (
        0x6627E8D5,
        0xE169C58D,
        0xBC57AC4C,
        0x9B00DBD8,
    )
    weights, ids = make_deepinfer_profile_routing(1, 10, 1, 0, torch.device("cpu"))
    expected_weight = torch.tensor(0x6627E8D5, dtype=torch.float32) * 2**-32 + 2**-33
    assert weights[0, 0] == expected_weight
    assert ids.tolist() == [[0xE169C58D % 10]]


def test_deepinfer_profile_preserves_shared_gate_weights_and_ids():
    """Random routed slots must not overwrite native sigmoid shared gates."""
    from vllm.model_executor.layers.fused_moe.router.deepinfer_profile import (
        make_deepinfer_profile_routing,
        replace_profile_routed_columns,
    )

    shared_logits = torch.tensor([[-3.0, 2.0], [0.0, 5.0]])
    shared_weights = shared_logits.sigmoid()
    shared_ids = torch.tensor([[10, 11], [10, 11]], dtype=torch.int32)
    native_weights = torch.cat((torch.ones(2, 2), shared_weights), dim=1)
    native_ids = torch.cat((torch.zeros(2, 2, dtype=torch.int32), shared_ids), dim=1)
    weights, ids = make_deepinfer_profile_routing(2, 10, 2, 0, torch.device("cpu"))
    actual_weights, actual_ids = replace_profile_routed_columns(
        native_weights, native_ids, weights, ids
    )
    assert torch.equal(actual_weights[:, :2], weights)
    assert torch.equal(actual_ids[:, :2], ids)
    assert torch.equal(actual_weights[:, 2:], shared_weights)
    assert torch.equal(actual_ids[:, 2:], shared_ids)
    assert torch.equal(shared_logits, torch.tensor([[-3.0, 2.0], [0.0, 5.0]]))


@pytest.fixture(scope="module")
def curand_profile_reference():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="deepinfer_profile_curand_reference",
        cpp_sources="""
        std::vector<torch::Tensor> profile_words(
            int64_t tokens, int64_t experts, int64_t k, int64_t rank);
        """,
        cuda_sources="""
        #include <torch/extension.h>
        #include <curand_kernel.h>
        #include <c10/cuda/CUDAStream.h>
        #include <c10/cuda/CUDAException.h>
        __global__ void generate(float* weights, int64_t* draws, int tokens,
                                 int experts, int k, uint64_t rank) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= tokens * k) return;
            int token = i / k, lane = i % k;
            curandStatePhilox4_32_10_t state;
            curand_init(rank, uint64_t(token) * 32 + lane, 0, &state);
            weights[i] = curand_uniform(&state);
            draws[i] = curand(&state) % (experts - lane);
        }
        std::vector<torch::Tensor> profile_words(
            int64_t tokens, int64_t experts, int64_t k, int64_t rank) {
            auto options = torch::TensorOptions().device(torch::kCUDA);
            auto weights = torch::empty({tokens, k}, options.dtype(torch::kFloat32));
            auto draws = torch::empty({tokens, k}, options.dtype(torch::kInt64));
            generate<<<(tokens * k + 127) / 128, 128, 0,
                       c10::cuda::getCurrentCUDAStream()>>>(
                weights.data_ptr<float>(), draws.data_ptr<int64_t>(),
                tokens, experts, k, rank);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return {weights, draws};
        }
        """,
        functions=["profile_words"],
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="NVIDIA cuRAND reference requires CUDA"
)
@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("top_k", [1, 8, 32])
def test_deepinfer_profile_matches_cuda_curand(curand_profile_reference, rank, top_k):
    """Check actual cuRAND seed/subsequence/word order, not just CPU Philox math."""
    from vllm.model_executor.layers.fused_moe.router.deepinfer_profile import (
        make_deepinfer_profile_routing,
    )

    weights, draws = curand_profile_reference.profile_words(17, 64, top_k, rank)
    expected_ids = []
    for row in draws.cpu().tolist():
        population = list(range(64))
        expected_ids.append([population.pop(draw) for draw in row])
    actual_weights, actual_ids = make_deepinfer_profile_routing(
        17, 64, top_k, rank, torch.device("cpu")
    )
    assert torch.equal(actual_weights, weights.cpu())
    assert actual_ids.tolist() == expected_ids
