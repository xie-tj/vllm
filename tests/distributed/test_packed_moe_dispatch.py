# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The real EP dispatcher preserves all four tensor payloads and row ordering."""

import os
from datetime import timedelta

import torch
import torch.multiprocessing as mp

from tests.utils import multi_gpu_test
from vllm.config import (
    KernelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    get_ep_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.parallel_state import graph_capture
from vllm.forward_context import get_forward_context, set_forward_context


def _worker(rank, store):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="8")
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=4,
            data_parallel_size=2,
            data_parallel_rank=rank // 4,
            enable_expert_parallel=True,
            is_moe_model=True,
            distributed_executor_backend="external_launcher",
        ),
        kernel_config=KernelConfig(enable_moe_dispatch_packing=True),
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=8,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"file://{store}",
            backend="nccl",
            timeout=timedelta(seconds=120),
        )
        initialize_model_parallel(4)
        try:
            group = get_ep_group()
            assert group.world_size == 8
            for first, second in ((1, 0), (4, 4), (16, 16), (64, 32), (128, 128)):
                sizes = [first] * 4 + [second] * 4
                rows = sizes[rank]
                torch.manual_seed(42 + rank)
                inputs = [
                    torch.randn(rows, 2048, device=device).to(torch.float8_e4m3fn),
                    torch.rand(rows, 8, device=device),
                    torch.randint(256, (rows, 8), device=device, dtype=torch.int32),
                    torch.rand(rows, 16, device=device),
                ]
                if rows:
                    inputs[1][0, 0] = float("nan")
                    inputs[3][0, 0] = float("inf")
                expected = group.all_gatherv(inputs, dim=0, sizes=sizes)

                def dispatch(rows=rows, first=first, second=second, inputs=inputs):
                    with set_forward_context(
                        None,
                        config,
                        num_tokens=rows * 4,
                        num_tokens_across_dp=torch.tensor([first * 4, second * 4]),
                    ):
                        metadata = get_forward_context().dp_metadata
                        assert metadata is not None
                        with metadata.sp_local_sizes(4, use_ep=True):
                            result = group.dispatch(
                                *inputs[:3],
                                is_sequence_parallel=True,
                                extra_tensors=[inputs[3]],
                            )
                    return [*result[:3], *result[3]]

                actual = dispatch()
                for left, right in zip(actual, expected, strict=True):
                    assert left.dtype == right.dtype and left.shape == right.shape
                    assert left.is_contiguous()
                    assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
                with graph_capture(device) as context:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=context.stream):
                        replayed = dispatch()
                graph.replay()
                torch.accelerator.synchronize()
                for left, right in zip(replayed, expected, strict=True):
                    assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
                del graph
        finally:
            cleanup_dist_env_and_memory()


@multi_gpu_test(num_gpus=8)
def test_packed_moe_dispatch_matches_native_payloads(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "store"),), nprocs=8, join=True)
