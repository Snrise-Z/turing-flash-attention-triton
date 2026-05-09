import argparse
import functools
import os
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv

from test_flash_attention_2080ti import _load_tutorial_module


@triton.jit
def _decode_attention_kernel(
    q,
    k,
    v,
    attention_mask,
    out,
    sm_scale,
    q_len: tl.constexpr,
    kv_len: tl.constexpr,
    head_dim: tl.constexpr,
    n_heads: tl.constexpr,
    mask_heads: tl.constexpr,
    mask_stride_b: tl.constexpr,
    mask_stride_h: tl.constexpr,
    mask_stride_q: tl.constexpr,
    mask_stride_k: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    query_idx = pid % q_len
    head_batch_idx = pid // q_len
    batch_idx = head_batch_idx // n_heads
    head_idx = head_batch_idx - batch_idx * n_heads
    mask_head_idx = tl.minimum(head_idx, mask_heads - 1)

    offs_d = tl.arange(0, head_dim)
    q_offset = (head_batch_idx * q_len + query_idx) * head_dim
    kv_offset = head_batch_idx * kv_len * head_dim
    out_offset = q_offset

    q_block = tl.load(q + q_offset + offs_d).to(tl.float32)
    q_abs_pos = kv_len - q_len + query_idx

    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros((head_dim,), dtype=tl.float32)

    for start_n in range(0, kv_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < kv_len
        causal_mask = offs_n <= q_abs_pos
        mask = kv_mask & causal_mask
        if HAS_MASK:
            mask_values = tl.load(
                attention_mask
                + batch_idx * mask_stride_b
                + mask_head_idx * mask_stride_h
                + query_idx * mask_stride_q
                + offs_n * mask_stride_k,
                mask=kv_mask,
                other=0,
            )
            mask = mask & mask_values

        k_block = tl.load(
            k + kv_offset + offs_n[:, None] * head_dim + offs_d[None, :],
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        qk = tl.sum(k_block * q_block[None, :], axis=1) * sm_scale * 1.4426950408889634
        qk = tl.where(mask, qk, -3.4028234663852886e38)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.where(mask, tl.exp2(qk - m_ij), 0.0)
        alpha = tl.exp2(m_i - m_ij)

        v_block = tl.load(
            v + kv_offset + offs_n[:, None] * head_dim + offs_d[None, :],
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v_block, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_ij

    acc = tl.where(l_i > 0.0, acc / l_i, 0.0)
    tl.store(out + out_offset + offs_d, acc)


def _normalize_attention_mask(
    attention_mask: torch.Tensor | None,
    batch: int,
    heads: int,
    q_len: int,
    kv_len: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim != 4:
        raise RuntimeError(f"local Triton backend requires a 4D attention mask, got {tuple(attention_mask.shape)}")
    if attention_mask.shape[0] != batch or attention_mask.shape[2] != q_len or attention_mask.shape[3] != kv_len:
        raise RuntimeError(
            "local Triton backend got incompatible attention mask shape "
            f"{tuple(attention_mask.shape)} for batch={batch}, q_len={q_len}, kv_len={kv_len}"
        )
    if attention_mask.shape[1] not in (1, heads):
        raise RuntimeError(
            "local Triton backend requires attention mask head dimension to be 1 or num_heads, "
            f"got {attention_mask.shape[1]} for num_heads={heads}"
        )
    if attention_mask.dtype == torch.bool:
        mask = attention_mask
    else:
        mask = attention_mask >= 0
    return mask.contiguous()


def _masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    mask = _normalize_attention_mask(attention_mask, query.shape[0], query.shape[1], query.shape[2], key.shape[2])
    if mask is None:
        raise RuntimeError("masked attention fallback requires a non-empty mask")
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1).masked_fill(~mask, 0.0)
    denom = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(denom > 0.0, probs / denom.clamp_min(torch.finfo(probs.dtype).tiny), probs)
    return torch.matmul(probs.to(query.dtype), value)


def _decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = torch.empty_like(query)
    batch, heads, q_len, head_dim = query.shape
    kv_len = key.shape[2]
    mask = _normalize_attention_mask(attention_mask, batch, heads, q_len, kv_len)
    if mask is None:
        mask_arg = query
        mask_heads = 1
        mask_stride_b = mask_stride_h = mask_stride_q = mask_stride_k = 0
    else:
        mask_arg = mask
        mask_heads = mask.shape[1]
        mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k = mask.stride()
    grid = (batch * heads * q_len,)
    _decode_attention_kernel[grid](
        query,
        key,
        value,
        mask_arg,
        output,
        float(scaling),
        q_len,
        kv_len,
        head_dim,
        heads,
        mask_heads,
        mask_stride_b,
        mask_stride_h,
        mask_stride_q,
        mask_stride_k,
        mask is not None,
        BLOCK_N=16,
        num_warps=4,
        num_stages=1,
    )
    return output


def register_triton_sm75_hd256_padded_backend() -> dict[str, int]:
    triton_mod = _load_tutorial_module(use_single_config=True, profile="sm75-hd256-16x16")
    calls = {"n": 0, "padded": 0, "decode": 0, "masked": 0}

    def triton_sm75_hd256_padded(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=None,
        is_causal=None,
        **kwargs,
    ):
        calls["n"] += 1
        if hasattr(module, "num_key_value_groups") and module.num_key_value_groups != 1:
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        if key.shape[2] != value.shape[2]:
            raise RuntimeError(
                "local Triton backend requires k/v to have the same sequence length, "
                f"got {key.shape[2]}/{value.shape[2]}"
            )

        n_ctx = query.shape[2]
        kv_len = key.shape[2]
        if n_ctx > kv_len:
            raise RuntimeError(f"local Triton backend got q_len > kv_len: {n_ctx}>{kv_len}")
        if n_ctx < kv_len:
            calls["decode"] += 1
            out = _decode_attention(
                query,
                key,
                value,
                float(scaling if scaling is not None else module.scaling),
                attention_mask=attention_mask,
            )
            return out.transpose(1, 2).contiguous(), None
        if attention_mask is not None:
            calls["masked"] += 1
            out = _masked_attention(
                query,
                key,
                value,
                attention_mask,
                float(scaling if scaling is not None else module.scaling),
            )
            return out.transpose(1, 2).contiguous(), None

        pad = (-n_ctx) % 16
        if pad:
            calls["padded"] += 1
            query = F.pad(query, (0, 0, 0, pad))
            key = F.pad(key, (0, 0, 0, pad))
            value = F.pad(value, (0, 0, 0, pad))

        out = triton_mod.attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            bool(is_causal if is_causal is not None else getattr(module, "is_causal", True)),
            float(scaling if scaling is not None else module.scaling),
            False,
        )
        out = out[:, :, :n_ctx, :]
        return out.transpose(1, 2).contiguous(), None

    ALL_MASK_ATTENTION_FUNCTIONS.register("triton_sm75_hd256_padded", sdpa_mask)
    ALL_ATTENTION_FUNCTIONS.register("triton_sm75_hd256_padded", triton_sm75_hd256_padded)
    return calls


def count_fla_calls(model) -> dict[str, int]:
    calls = {"chunk": 0, "recurrent": 0}
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3_5GatedDeltaNet":
            continue

        orig_chunk = module.chunk_gated_delta_rule
        orig_recurrent = module.recurrent_gated_delta_rule

        @functools.wraps(orig_chunk)
        def chunk_wrapper(*args, __orig=orig_chunk, **kwargs):
            calls["chunk"] += 1
            return __orig(*args, **kwargs)

        @functools.wraps(orig_recurrent)
        def recurrent_wrapper(*args, __orig=orig_recurrent, **kwargs):
            calls["recurrent"] += 1
            return __orig(*args, **kwargs)

        module.chunk_gated_delta_rule = chunk_wrapper
        module.recurrent_gated_delta_rule = recurrent_wrapper
    return calls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen3.5-9B 4-bit on a single sm75 GPU with FLA linear attention and local Triton full attention."
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/data/huggingface-z/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    )
    parser.add_argument("--prompt", default="VLN是什么？")
    parser.add_argument("--generate", action="store_true", help="Run generation after the forward smoke test.")
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--no-cache", action="store_true", help="Disable KV cache during generation.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(
        f"device={torch.cuda.get_device_name(0)} capability=sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}",
        flush=True,
    )

    full_attention_calls = register_triton_sm75_hd256_padded_backend()
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    start = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=qconfig,
        device_map={"": 0},
        dtype=torch.float16,
        attn_implementation="triton_sm75_hd256_padded",
    ).eval()
    print(f"loaded_s={time.perf_counter() - start:.3f}", flush=True)

    fla_calls = count_fla_calls(model)
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    print(f"input_tokens={inputs['input_ids'].shape[-1]}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False, logits_to_keep=1)
    torch.cuda.synchronize()
    print(
        f"MODEL_FORWARD_OK logits_shape={tuple(outputs.logits.shape)} forward_s={time.perf_counter() - start:.3f}",
        flush=True,
    )

    if args.generate:
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=not args.no_cache,
            )
        torch.cuda.synchronize()
        print(
            f"GENERATE_OK tokens={generated.shape[-1]} generate_s={time.perf_counter() - start:.3f}",
            flush=True,
        )
        print(tokenizer.decode(generated[0], skip_special_tokens=True), flush=True)

    print(f"fla_calls={fla_calls}", flush=True)
    print(f"full_attention_calls={full_attention_calls}", flush=True)
    print(f"max_alloc_gib={torch.cuda.max_memory_allocated() / 1024**3:.3f}", flush=True)


if __name__ == "__main__":
    main()
