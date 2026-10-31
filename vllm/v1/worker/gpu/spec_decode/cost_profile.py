# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup forward measurements, isolated from serving graphs and routing."""

from collections.abc import Callable

import torch
import torch.distributed as dist

from vllm.compilation.cudagraph_pool import capture_pool
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_ep_group, get_tp_group
from vllm.model_executor.layers.fused_moe.router.deepinfer_profile import (
    make_deepinfer_profile_routing,
)
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.offloader.base import get_offloader
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.worker.gpu.async_utils import StepTimingSample
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CreateForwardFn,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers


class DecodeCostProfiler:
    """Capture profiling graphs first; measure only after all capture finishes."""

    def __init__(
        self,
        model: torch.nn.Module,
        input_buffers: InputBuffers,
        vllm_config: VllmConfig,
        *,
        moe: bool,
    ) -> None:
        self.input_buffers = input_buffers
        self.vllm_config = vllm_config
        self.group = get_ep_group() if moe else get_tp_group()
        self.runners = (
            [m for m in model.modules() if isinstance(m, MoERunner)] if moe else []
        )
        if moe and not self.runners:
            raise ValueError("Online MoE profiling requires a standard MoE runner.")
        self.routes: dict[MoERunner, tuple[torch.Tensor, torch.Tensor]] = {}
        tables = {}
        max_rows = len(input_buffers.input_ids) * self.group.world_size
        for runner in self.runners:
            if runner.routed_experts.quant_method.is_monolithic or isinstance(
                runner.router, ZeroExpertRouter
            ):
                raise ValueError(
                    "Exact startup random routing requires a native physical-ID "
                    "routing entry. The selected "
                    f"{type(runner.routed_experts.quant_method).__name__}/"
                    f"{type(runner.router).__name__} does not expose one; "
                    "its backend is not changed automatically."
                )
            parallel = runner.moe_config.moe_parallel_config
            # Gathered tensor-parallel token streams need the same selections
            # on all shards. Local EP streams use their EP rank.
            rank = parallel.ep_rank if parallel.use_ep else 0
            if runner.do_naive_dispatch_combine:
                rank = 0
            experts = runner.routed_experts.global_num_experts
            top_k = runner.routed_experts.top_k
            dtype = runner.routed_experts.quant_method.topk_indices_dtype or torch.int32
            key = experts, top_k, rank, dtype
            if key not in tables:
                tables[key] = make_deepinfer_profile_routing(
                    max_rows, experts, top_k, rank, input_buffers.device, dtype
                )
            self.routes[runner] = tables[key]
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict[
            BatchExecutionDescriptor,
            tuple[
                torch.cuda.CUDAGraph,
                CreateForwardFn,
                Callable[[CUDAGraphMode], None],
            ],
        ] = {}
        self.samples: list[StepTimingSample] = []

    def capture(
        self, desc: BatchExecutionDescriptor, create_forward_fn: CreateForwardFn
    ) -> None:
        """Record a profiling graph without measuring during graph capture."""
        previous = [r.profile_random_routing for r in self.runners]
        try:
            for runner in self.runners:
                runner.profile_random_routing = self.routes[runner]
            with torch.random.fork_rng(devices=[self.input_buffers.device]):
                forward = create_forward_fn(desc, warmup=True)
                self.input_buffers.is_padding[: desc.num_tokens].fill_(False)
                forward(CUDAGraphMode.NONE)
                get_offloader().sync_prev_onload()
                forward = create_forward_fn(desc, warmup=False)
                self.input_buffers.is_padding[: desc.num_tokens].fill_(False)
                graph = torch.cuda.CUDAGraph()
                with (
                    capture_pool(self.pool, self.vllm_config) as pool,
                    torch.cuda.graph(
                        graph,
                        pool=pool,
                        stream=torch.cuda.current_stream(self.input_buffers.device),
                    ),
                ):
                    forward(CUDAGraphMode.NONE)
                    get_offloader().join_after_forward()
                # The closure owns metadata/input tensors allocated outside
                # the graph pool. Keep them alive until all replays finish.
                self.graphs[desc] = graph, create_forward_fn, forward
        finally:
            for runner, original in zip(self.runners, previous):
                runner.profile_random_routing = original

    def measure(self) -> None:
        """Run 1 warmup + 5 timed replays, largest first, outside capture/JIT."""
        if self.samples:
            raise RuntimeError("Startup cost graphs have already been measured")
        device = self.input_buffers.device
        with torch.random.fork_rng(devices=[device]), gpu_sync_allowed():
            torch.accelerator.synchronize()
            sync = torch.zeros(1, device=device)
            for desc in sorted(self.graphs, key=lambda d: d.num_tokens, reverse=True):
                graph, prepare, _captured_forward = self.graphs[desc]
                # Restore each shape's shared input buffers without a model run.
                prepare(desc, warmup=False)
                self.input_buffers.is_padding[: desc.num_tokens].fill_(False)
                graph.replay()
                dist.all_reduce(sync, group=self.group.device_group)
                events = []
                for _ in range(5):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    dist.all_reduce(sync, group=self.group.device_group)
                    events.append((start, end))
                for start, end in events:
                    end.synchronize()
                    self.samples.append(
                        StepTimingSample(
                            start.elapsed_time(end),
                            0.0,
                            desc.num_tokens,
                            desc.num_reqs or desc.num_tokens,
                            True,
                        )
                    )
            torch.accelerator.synchronize()
