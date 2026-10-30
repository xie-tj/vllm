# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One cost scope and budget decision for a cooperating MoE TP x DP group."""

from collections import Counter, deque
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.distributed.parallel_state import get_ep_group, get_tp_group
from vllm.v1.worker.gpu.async_utils import RuntimeTimingContext

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.async_utils import RuntimeTimingCollector

    from .adaptive_verification import OnlineCostModel


@dataclass(frozen=True)
class BudgetProposal:
    num_reqs: int
    base_tokens: int
    accepted: np.ndarray
    eligible: bool = True


def select_parallel_budgets(
    proposals: list[BudgetProposal], model: "OnlineCostModel"
) -> tuple[tuple[int, ...], int, float]:
    """Maximize group acceptance at the physical shape selected by DP padding."""
    base = np.array([p.base_tokens for p in proposals])
    sizes = model.graph_sizes[model.graph_sizes >= base.max()]
    if not len(sizes):
        return tuple(0 for _ in proposals), 0, 0.0
    capacity = np.array([len(p.accepted) - 1 for p in proposals])
    budgets = np.minimum(sizes[:, None] - base, capacity)
    physical = model.graph_sizes[
        np.searchsorted(model.graph_sizes, (budgets + base).max(axis=1))
    ]
    costs = model.get_parallel_costs([p.num_reqs for p in proposals], physical)
    accepted = np.zeros(len(sizes))
    for rank, proposal in enumerate(proposals):
        accepted += proposal.accepted[budgets[:, rank]]
    best = int(np.argmax(accepted / costs))
    return tuple(int(b) for b in budgets[best]), int(physical[best]), float(costs[best])


@dataclass(frozen=True)
class ParallelCostDecision:
    budgets: tuple[int, ...] | None
    context: RuntimeTimingContext


class ParallelCostCoordinator:
    """All ranks call once per target step, including idle and profiler steps.

    Only each DP replica's TP leader contributes acceptance estimates. The group
    driver learns completed timings and publishes the entire budget vector.
    """

    def __init__(self, max_num_tokens: int):
        self.group = get_ep_group()
        self.tp_size = get_tp_group().world_size
        self.dp_size = self.group.world_size // self.tp_size
        self.max_num_tokens = max_num_tokens
        self.step_id = 0
        self.epoch = 0
        self.cost_version = 0
        self._profile_mask: tuple[bool, ...] = ()
        self.decision: ParallelCostDecision | None = None
        self.elapsed_ms = 0.0
        self.budgets: list[Counter[int]] = [Counter() for _ in range(self.dp_size)]
        self.shapes: Counter[int] = Counter()
        self.observations: deque[dict] = deque(maxlen=256)

    def coordinate(
        self,
        proposal: BudgetProposal,
        model: "OnlineCostModel | None",
        timing: "RuntimeTimingCollector | None",
        *,
        profiling: bool,
    ) -> ParallelCostDecision:
        start = perf_counter()
        self.step_id += 1
        rank = self.group.rank_in_group
        tensor = torch.zeros(
            self.group.world_size, self.max_num_tokens + 7, dtype=torch.float64
        )
        tensor[rank, :6] = torch.tensor(
            [
                self.step_id,
                profiling,
                proposal.eligible,
                proposal.num_reqs,
                proposal.base_tokens,
                len(proposal.accepted),
            ]
        )
        if get_tp_group().is_first_rank:
            tensor[rank, 6 : 6 + len(proposal.accepted)] = torch.from_numpy(
                proposal.accepted
            )
        self.group.cpu_all_reduce_(tensor)
        assert bool(torch.all(tensor[:, 0] == self.step_id)), "MoE cost step mismatch"
        profile_mask = tuple(bool(v) for v in tensor[:, 1])
        if profile_mask != self._profile_mask:
            self.epoch += 1
            self._profile_mask = profile_mask
            if timing is not None:
                timing.invalidate()
        leaders = tensor[:: self.tp_size]
        proposals = [
            BudgetProposal(
                int(row[3]),
                int(row[4]),
                row[6 : 6 + int(row[5])].numpy().copy(),
                bool(row[2]),
            )
            for row in leaders
        ]
        # Followers may enter profiling independently; every vote counts.
        eligible = (
            not any(profile_mask)
            and bool(torch.all(tensor[:, 2] == 1))
            and any(p.num_reqs for p in proposals)
        )
        decision = None
        if self.group.is_first_rank:
            assert model is not None and timing is not None
            timing.cancel()
            samples = [
                s
                for s in timing.collect_ready()
                if (s.context is not None and s.context.epoch == self.epoch)
            ]
            if samples and not any(profile_mask):
                accepted = model.update(samples)
                self.cost_version += bool(accepted)
                for sample in accepted:
                    self.observations.append(
                        {
                            "context": sample.context,
                            "forward_ms": sample.forward_ms,
                            "sampler_ms": sample.sampler_ms,
                            "num_target_tokens": sample.num_target_tokens,
                            "num_reqs": sample.num_reqs,
                        }
                    )
            budgets, physical, predicted = (None, 0, 0.0)
            if eligible:
                budgets, physical, predicted = select_parallel_budgets(proposals, model)
                if physical == 0:
                    budgets = None
            context = RuntimeTimingContext(
                step_id=self.step_id,
                epoch=self.epoch,
                cost_version=self.cost_version,
                num_reqs_across_dp=tuple(p.num_reqs for p in proposals),
                num_tokens_across_dp=(physical,) * self.dp_size,
                num_valid_tokens_across_dp=tuple(
                    p.base_tokens + b if p.num_reqs else 0
                    for p, b in zip(proposals, budgets or (0,) * self.dp_size)
                ),
                predicted_ms=predicted,
                group_ranks=tuple(self.group.ranks),
                predicted_driver_ms=(
                    float(
                        model.get_costs(proposals[0].num_reqs, np.array([physical]))[0]
                    )
                    if physical and proposals[0].num_reqs
                    else 0.0
                ),
            )
            decision = ParallelCostDecision(budgets, context)
        decision = self.group.broadcast_object(decision, src=0)
        assert isinstance(decision, ParallelCostDecision)
        assert decision.context.step_id == self.step_id
        assert decision.context.epoch == self.epoch
        self.cost_version = decision.context.cost_version
        self.decision = decision
        if decision.budgets is not None:
            for counts, budget in zip(self.budgets, decision.budgets):
                counts[budget] += 1
            self.shapes[decision.context.num_tokens_across_dp[0]] += 1
        self.elapsed_ms += (perf_counter() - start) * 1000
        return decision

    def diagnostics(self) -> dict:
        """Bounded serving evidence, available through worker collective RPC."""
        return {
            "group_ranks": self.group.ranks,
            "driver": self.group.is_first_rank,
            "steps": self.step_id,
            "epoch": self.epoch,
            "cost_version": self.cost_version,
            "coordination_ms": self.elapsed_ms,
            "budgets": [sorted(c.items()) for c in self.budgets],
            "physical_shapes": sorted(self.shapes.items()),
            "observations": [
                dict(value, context=asdict(value["context"]))
                for value in self.observations
            ],
        }
