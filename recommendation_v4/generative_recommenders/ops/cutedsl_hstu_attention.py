# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict

import functools
import os
from typing import Any, Callable, Optional, Tuple

import torch

_TRUE_VALUES = {"1", "true", "yes", "on"}
_DLPACK_DIM_LIMIT = 1 << 31
_WORKSPACE_INNER_DIM = 128
_LARGE_WORKSPACE_BYTES = 1 << 33


def cutedsl_hstu_enabled() -> bool:
    return os.getenv("ENABLE_CUTEDSL_HSTU", "0").strip().lower() in _TRUE_VALUES


def _patch_large_workspace_dlpack(hstu_ops_gpu: Any) -> None:
    original_from_dlpack = hstu_ops_gpu.from_dlpack
    if getattr(original_from_dlpack, "_hstu_large_workspace_compatible", False):
        return

    @functools.wraps(original_from_dlpack)
    def compatible_from_dlpack(tensor: Any, *args: Any, **kwargs: Any) -> Any:
        # CUTLASS DSL uses signed 32-bit memref dimensions. The HSTU backward
        # workspace is byte-addressed, so an equivalent 2-D view avoids a
        # dimension overflow without changing its storage or kernel iterator.
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.dtype == torch.uint8
            and tensor.ndim == 1
            and tensor.numel() >= _DLPACK_DIM_LIMIT
        ):
            if tensor.numel() % _WORKSPACE_INNER_DIM != 0:
                raise RuntimeError("CUTEDSL HSTU workspace is not 128-byte aligned")
            tensor = tensor.view(-1, _WORKSPACE_INNER_DIM)
        return original_from_dlpack(tensor, *args, **kwargs)

    compatible_from_dlpack._hstu_large_workspace_compatible = True
    hstu_ops_gpu.from_dlpack = compatible_from_dlpack


@functools.lru_cache(maxsize=1)
def _load_cutedsl_hstu_ops() -> Tuple[Callable[..., Any], Callable[..., Any]]:
    try:
        from hstu.hstu_blackwell import hstu_ops_gpu
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "ENABLE_CUTEDSL_HSTU=1 requires the Blackwell hstu package and a "
            "compatible nvidia-cutlass-dsl installation. The validated "
            "distributed-recommender:devel_latest setup uses "
            "nvidia-cutlass-dsl==4.4.2."
        ) from error
    _patch_large_workspace_dlpack(hstu_ops_gpu)
    return hstu_ops_gpu.hstu_varlen_fwd_100, hstu_ops_gpu.hstu_varlen_bwd_100


class _CuteDslHstuAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        num_targets: Optional[torch.Tensor],
        max_seq_len: int,
        alpha: float,
        max_attn_len: int,
    ) -> torch.Tensor:
        hstu_fwd, _ = _load_cutedsl_hstu_ops()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        seq_offsets = seq_offsets.to(dtype=torch.int32).contiguous()
        has_num_targets = num_targets is not None
        targets = (
            num_targets.to(dtype=torch.int32).contiguous()
            if num_targets is not None
            else torch.empty(0, dtype=torch.int32, device=q.device)
        )
        window_size_left = max_attn_len if max_attn_len > 0 else -1
        try:
            output, _ = hstu_fwd(
                q,
                k,
                v,
                seq_offsets,
                seq_offsets,
                max_seq_len,
                max_seq_len,
                None,
                targets if has_num_targets else None,
                1,
                window_size_left,
                0,
                alpha,
                None,
                None,
            )
        except AttributeError as error:
            raise RuntimeError(
                "The installed hstu and nvidia-cutlass-dsl packages are API "
                "incompatible. The validated devel_latest setup uses "
                "nvidia-cutlass-dsl==4.4.2."
            ) from error
        ctx.save_for_backward(q, k, v, seq_offsets, targets)
        ctx.has_num_targets = has_num_targets
        ctx.max_seq_len = max_seq_len
        ctx.alpha = alpha
        ctx.window_size_left = window_size_left
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
    ]:
        _, hstu_bwd = _load_cutedsl_hstu_ops()
        q, k, v, seq_offsets, targets = ctx.saved_tensors
        dq, dk, dv, _ = hstu_bwd(
            grad_output.contiguous(),
            q,
            k,
            v,
            seq_offsets,
            seq_offsets,
            ctx.max_seq_len,
            ctx.max_seq_len,
            None,
            None,
            None,
            None,
            targets if ctx.has_num_targets else None,
            1,
            ctx.window_size_left,
            0,
            ctx.alpha,
            None,
            False,
            None,
            False,
        )
        workspace_bytes = (
            ctx.max_seq_len
            * q.shape[1]
            * q.shape[2]
            * (seq_offsets.numel() - 1)
            * 4
        )
        if (
            workspace_bytes >= _LARGE_WORKSPACE_BYTES
            and os.getenv("NCCL_DMABUF_ENABLE", "0").strip().lower()
            in _TRUE_VALUES
        ):
            # The CUTEDSL dQ workspace has no live references after hstu_bwd,
            # but its cached 8+ GiB segment is invisible to NCCL's allocator.
            # Return that free segment to CUDA before TorchRec's sparse
            # backward allocates its DMABUF communication buffers.
            torch.cuda.empty_cache()
        return dq, dk, dv, None, None, None, None, None


def cutedsl_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool,
    dropout_pr: float,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    min_full_attn_seq_len: int,
) -> torch.Tensor:
    if not q.is_cuda:
        raise RuntimeError("CUTEDSL HSTU attention requires CUDA tensors")
    capability = torch.cuda.get_device_capability(q.device)
    if capability != (10, 0):
        raise RuntimeError(
            "CUTEDSL HSTU attention currently requires an SM100 Blackwell GPU; "
            f"got compute capability {capability[0]}.{capability[1]}"
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("CUTEDSL HSTU attention supports only bf16 and fp16")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("CUTEDSL HSTU attention requires identical Q, K, and V shapes")
    if q.shape[-1] not in (64, 128):
        raise ValueError(
            "CUTEDSL HSTU backward supports head dimensions 64 and 128 only"
        )
    if not causal:
        raise ValueError("CUTEDSL HSTU attention supports only causal attention")
    if dropout_pr >= 1e-6:
        raise ValueError("CUTEDSL HSTU attention does not support dropout")
    if attn_scale is not None:
        raise ValueError("CUTEDSL HSTU attention does not support attn_scale")
    if contextual_seq_len != 0:
        raise ValueError("CUTEDSL HSTU attention does not support contextual masking")
    if min_full_attn_seq_len != 0:
        raise ValueError("CUTEDSL HSTU attention does not support full-attention tails")

    return _CuteDslHstuAttention.apply(
        q,
        k,
        v,
        seq_offsets,
        num_targets,
        max_seq_len,
        alpha,
        max_attn_len,
    )
