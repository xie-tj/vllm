# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stateless physical routing used only by startup cost profiling.

Matches DeepInfer's pinned DeepGEMM store_random_topk_token protocol
(1fc9c66975ee322d628b3288534ca0e9ce12a64d). Materialize the deterministic
rank/token/lane tables on the host, outside graph capture and measurement.
"""

import numpy as np
import torch


def make_deepinfer_profile_routing(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    rank: int,
    device: torch.device,
    indices_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return random positive weights and distinct physical expert IDs per token."""
    if not 0 < top_k <= min(num_experts, 32) or num_tokens < 0 or rank < 0:
        raise ValueError("Invalid DeepInfer profiling routing dimensions or rank")
    lanes = np.arange(top_k, dtype=np.uint64)[None, :]
    sequence = np.arange(num_tokens, dtype=np.uint64)[:, None] * 32 + lanes
    mask = np.uint64(0xFFFFFFFF)
    # curand_init(seed=rank, subsequence=token*32+lane, offset=0).
    c0 = np.zeros_like(sequence)
    c1 = np.zeros_like(sequence)
    c2, c3 = sequence & mask, sequence >> 32
    k0, k1 = np.uint64(rank & 0xFFFFFFFF), np.uint64(rank >> 32)
    for _ in range(10):
        p0 = np.uint64(0xD2511F53) * c0
        p1 = np.uint64(0xCD9E8D57) * c2
        c0, c1, c2, c3 = (
            ((p1 >> 32) ^ c1 ^ k0) & mask,
            p1 & mask,
            ((p0 >> 32) ^ c3 ^ k1) & mask,
            p0 & mask,
        )
        k0 = (k0 + np.uint64(0x9E3779B9)) & mask
        k1 = (k1 + np.uint64(0xBB67AE85)) & mask
    weights = c0.astype(np.float32) * np.float32(2**-32) + np.float32(2**-33)
    candidates = c1 % (np.uint64(num_experts) - lanes)
    selected = np.empty_like(candidates)
    rows = np.arange(num_tokens)
    for _ in range(top_k):
        winner = np.argmin(candidates, axis=1)
        minimum = candidates[rows, winner].copy()
        selected[rows, winner] = minimum
        candidates[rows, winner] = mask
        candidates += (
            (candidates != mask)
            & (candidates >= minimum[:, None])
            & (lanes > winner[:, None])
        ).astype(np.uint64)
    return (
        torch.from_numpy(weights).to(device=device),
        torch.from_numpy(selected.astype(np.int64)).to(
            device=device, dtype=indices_dtype
        ),
    )


def replace_profile_routed_columns(
    native_weights: torch.Tensor,
    native_ids: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace routed selections while retaining every native shared slot."""
    top_k = ids.shape[1]
    if native_ids.shape[1] <= top_k or native_weights.shape[1] <= top_k:
        raise ValueError("The selected router did not provide its fused shared slots")
    native_weights[:, :top_k].copy_(weights)
    native_ids[:, :top_k].copy_(ids)
    return native_weights, native_ids
