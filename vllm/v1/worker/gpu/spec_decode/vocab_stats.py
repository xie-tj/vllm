# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exchange exact vocabulary-block statistics instead of full draft logits."""

import torch

from vllm.distributed import get_tp_group
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    VOCAB_BLOCK_SIZE,
    compute_acceptance_partials,
)


@triton.jit
def _pack_blocks(
    output,
    logits,
    edges,
    row_stride,
    num_rows,
    local_width,
    VOCAB: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    FIRST: tl.constexpr,
    OWNED: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
    columns = (FIRST + block) * BLOCK + tl.arange(0, BLOCK)
    valid = (block < OWNED) & (columns < VOCAB)
    local_column = columns - RANK * local_width
    local = valid & (local_column >= 0) & (local_column < local_width)
    value = tl.load(
        logits + row * row_stride + local_column, mask=local, other=-float("inf")
    )
    edge_column = columns - (RANK + 1) * local_width
    edge = tl.load(
        edges + (((RANK + 1) % WORLD) * num_rows + row) * BLOCK + edge_column,
        mask=valid & ~local,
        other=-float("inf"),
    )
    tl.store(
        output + (row * MAX_BLOCKS + block) * BLOCK + tl.arange(0, BLOCK),
        tl.where(local, value, edge),
    )


@triton.jit
def _unpack_blocks(
    packed,
    maxima,
    sumexp,
    sampled,
    local_width,
    VOCAB: tl.constexpr,
    WORLD: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    GLOBAL_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    PADDED: tl.constexpr,
):
    row = tl.program_id(0)
    blocks = tl.arange(0, PADDED)
    mask = blocks < GLOBAL_BLOCKS
    owners = tl.where(mask, (blocks * BLOCK) // local_width, 0)
    local_block = blocks - tl.cdiv(owners * local_width, BLOCK)
    pitch = 4 * MAX_BLOCKS
    base = row * WORLD * pitch + owners * pitch + local_block
    m = tl.load(packed + base, mask=mask, other=-float("inf"))
    s = tl.load(packed + base + MAX_BLOCKS, mask=mask, other=0.0)
    tl.store(maxima + row * GLOBAL_BLOCKS + blocks, m, mask=mask)
    tl.store(sumexp + row * GLOBAL_BLOCKS + blocks, s, mask=mask)
    raw_max = tl.load(packed + base + 2 * MAX_BLOCKS, mask=mask, other=-float("inf"))
    raw_index = tl.load(packed + base + 3 * MAX_BLOCKS, mask=mask, other=VOCAB).to(
        tl.int32
    )
    nan = mask & (raw_max != raw_max)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(nan, -float("inf"), raw_max), 0)
    winner = mask & tl.where(has_nan, nan, raw_max == maximum)
    index = tl.min(tl.where(winner, raw_index, VOCAB), 0)
    tl.store(sampled + row, index)


def gather_draft_statistics(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return greedy ids and partials in the original global block order.

    Boundary columns are exchanged before reducing, so a vocabulary block split
    across TP shards has exactly the same reduction as the full-logit path.
    The score remains a prefix-only log-odds, independent of the sampled token.
    """
    group = get_tp_group()
    rows, width = logits.shape
    block = VOCAB_BLOCK_SIZE
    if not 0 < vocab_size <= width * group.world_size:
        raise ValueError("Vocabulary size exceeds the provided TP shards")
    if group.world_size == 1 or width < block or rows == 0:
        full = group.all_gather(logits, dim=-1) if group.world_size > 1 else logits
        full = full[:, :vocab_size]
        return full.argmax(-1), *compute_acceptance_partials(
            full, idx_mapping, temperature
        )
    if logits.stride(1) != 1 or vocab_size >= 2**24:
        raise ValueError(
            "Vocabulary statistics require contiguous columns and ids below 2**24"
        )
    rank, world = group.rank_in_group, group.world_size
    first = triton.cdiv(rank * width, block)
    owned = triton.cdiv((rank + 1) * width, block) - first
    max_blocks = triton.cdiv(width, block) + 1
    edges = group.all_gather(logits[:, :block].contiguous(), dim=0)
    dense = torch.empty(
        rows, max_blocks * block, dtype=logits.dtype, device=logits.device
    )
    _pack_blocks[(rows, max_blocks)](
        dense,
        logits,
        edges,
        logits.stride(0),
        rows,
        width,
        VOCAB=vocab_size,
        RANK=rank,
        WORLD=world,
        FIRST=first,
        OWNED=owned,
        MAX_BLOCKS=max_blocks,
        BLOCK=block,
    )
    maxima, sumexp = compute_acceptance_partials(dense, idx_mapping, temperature)
    raw_max, raw_index = dense.view(rows, max_blocks, block).max(-1)
    global_blocks = torch.arange(max_blocks, device=logits.device) + first
    raw_index = raw_index + global_blocks * block
    packed = torch.cat((maxima, sumexp, raw_max.float(), raw_index.float()), dim=1)
    packed = group.all_gather(packed, dim=-1)
    count = triton.cdiv(vocab_size, block)
    full_max = torch.empty(rows, count, dtype=torch.float32, device=logits.device)
    full_sum = torch.empty_like(full_max)
    sampled = torch.empty(rows, dtype=torch.int64, device=logits.device)
    _unpack_blocks[(rows,)](
        packed,
        full_max,
        full_sum,
        sampled,
        width,
        VOCAB=vocab_size,
        WORLD=world,
        MAX_BLOCKS=max_blocks,
        GLOBAL_BLOCKS=count,
        BLOCK=block,
        PADDED=triton.next_power_of_2(count),
    )
    return sampled, full_max, full_sum
