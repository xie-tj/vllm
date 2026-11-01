# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless row-wise packing for modular MoE dispatch tensors."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _pack_bytes(
    out,
    a,
    b,
    c,
    d,
    rows,
    A: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    width = A + B + C + D
    row, col = offset // width, offset % width
    valid = row < rows
    va = tl.load(a + row * A + col, valid & (col < A), other=0)
    vb = tl.load(b + row * B + col - A, valid & (col >= A) & (col < A + B), other=0)
    vc = tl.load(
        c + row * C + col - A - B, valid & (col >= A + B) & (col < A + B + C), other=0
    )
    vd = tl.load(d + row * D + col - A - B - C, valid & (col >= A + B + C), other=0)
    value = tl.where(
        col < A, va, tl.where(col < A + B, vb, tl.where(col < A + B + C, vc, vd))
    )
    tl.store(out + offset, value, valid)


@triton.jit
def _unpack_bytes(
    src,
    a,
    b,
    c,
    d,
    rows,
    A: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    width = A + B + C + D
    row, col = offset // width, offset % width
    valid = row < rows
    value = tl.load(src + offset, valid, other=0)
    tl.store(a + row * A + col, value, valid & (col < A))
    tl.store(b + row * B + col - A, value, valid & (col >= A) & (col < A + B))
    tl.store(
        c + row * C + col - A - B, value, valid & (col >= A + B) & (col < A + B + C)
    )
    tl.store(d + row * D + col - A - B - C, value, valid & (col >= A + B + C))


def _pack_rows(tensors: list[torch.Tensor]) -> torch.Tensor:
    assert len(tensors) == 4
    rows = tensors[0].shape[0]
    assert all(
        t.ndim == 2 and t.shape[0] == rows and t.is_contiguous() for t in tensors
    )
    views = [t.view(torch.uint8) for t in tensors]
    widths = [t.shape[1] for t in views]
    packed = torch.empty(rows, sum(widths), dtype=torch.uint8, device=tensors[0].device)
    if rows:
        _pack_bytes[(triton.cdiv(packed.numel(), 1024),)](
            packed, *views, rows, *widths, BLOCK=1024, num_warps=4
        )
    return packed


def _unpack_rows(
    packed: torch.Tensor, templates: list[torch.Tensor]
) -> list[torch.Tensor]:
    rows = packed.shape[0]
    result = [
        torch.empty(rows, t.shape[1], dtype=t.dtype, device=t.device) for t in templates
    ]
    views = [t.view(torch.uint8) for t in result]
    widths = [t.shape[1] for t in views]
    assert sum(widths) == packed.shape[1]
    if rows:
        _unpack_bytes[(triton.cdiv(packed.numel(), 1024),)](
            packed, *views, rows, *widths, BLOCK=1024, num_warps=4
        )
    return result


def packed_all_gatherv(
    tensors: list[torch.Tensor], group, sizes: list[int]
) -> list[torch.Tensor]:
    """Gather four opaque row tensors, preserving every byte and output stride."""
    contiguous = [tensor.contiguous() for tensor in tensors]
    packed = _pack_rows(contiguous)
    gathered = group.all_gatherv(packed, dim=0, sizes=sizes)
    return _unpack_rows(gathered, contiguous)
