import argparse
import functools
import math
import os
import time

import torch
import xformers
import xformers.ops as xops
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
from xformers.ops import fmha


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
        raise RuntimeError(f"xFormers backend requires a 4D attention mask, got {tuple(attention_mask.shape)}")
    if attention_mask.shape[0] != batch or attention_mask.shape[2] != q_len or attention_mask.shape[3] != kv_len:
        raise RuntimeError(
            "xFormers backend got incompatible attention mask shape "
            f"{tuple(attention_mask.shape)} for batch={batch}, q_len={q_len}, kv_len={kv_len}"
        )
    if attention_mask.shape[1] not in (1, heads):
        raise RuntimeError(
            "xFormers backend requires attention mask head dimension to be 1 or num_heads, "
            f"got {attention_mask.shape[1]} for num_heads={heads}"
        )
    if attention_mask.dtype == torch.bool:
        return attention_mask
    return attention_mask >= 0


def _make_aligned_xformers_bias(
    attention_mask: torch.Tensor,
    batch: int,
    heads: int,
    q_len: int,
    kv_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    mask = _normalize_attention_mask(attention_mask, batch, heads, q_len, kv_len)
    if mask is None:
        raise RuntimeError("aligned xFormers bias requires a non-empty mask")

    kv_aligned = math.ceil(kv_len / 8) * 8
    min_value = torch.finfo(dtype).min
    storage = torch.full((batch, heads, q_len, kv_aligned), min_value, dtype=dtype, device=device)
    visible = mask.expand(batch, heads, q_len, kv_len)
    storage[..., :kv_len].copy_(torch.where(visible, torch.zeros((), dtype=dtype, device=device), min_value))
    return storage[..., :kv_len]


def _visible_query_rows(attention_mask: torch.Tensor, batch: int, heads: int, q_len: int, kv_len: int) -> torch.Tensor:
    mask = _normalize_attention_mask(attention_mask, batch, heads, q_len, kv_len)
    if mask is None:
        raise RuntimeError("visible query rows require a non-empty mask")
    return mask.expand(batch, heads, q_len, kv_len).any(dim=-1)


def register_xformers_memory_efficient_backend() -> dict[str, int]:
    calls = {"n": 0, "prefill": 0, "decode": 0, "masked": 0, "aligned_bias": 0}
    bottom_right_cls = getattr(fmha.attn_bias, "LowerTriangularFromBottomRightMask", None)

    def xformers_memory_efficient_aligned(
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
            raise RuntimeError(f"xFormers backend requires k/v length match, got {key.shape[2]}/{value.shape[2]}")

        batch, heads, q_len, _ = query.shape
        kv_len = key.shape[2]
        if q_len > kv_len:
            raise RuntimeError(f"xFormers backend got q_len > kv_len: {q_len}>{kv_len}")

        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        scale = float(scaling if scaling is not None else module.scaling)

        if q_len < kv_len:
            calls["decode"] += 1
        else:
            calls["prefill"] += 1

        if attention_mask is not None:
            calls["masked"] += 1
            calls["aligned_bias"] += 1
            attn_bias = _make_aligned_xformers_bias(
                attention_mask,
                batch=batch,
                heads=heads,
                q_len=q_len,
                kv_len=kv_len,
                dtype=query.dtype,
                device=query.device,
            )
            visible_rows = _visible_query_rows(attention_mask, batch, heads, q_len, kv_len)
        elif bool(is_causal if is_causal is not None else getattr(module, "is_causal", True)):
            visible_rows = None
            if q_len == kv_len:
                attn_bias = fmha.attn_bias.LowerTriangularMask()
            else:
                if bottom_right_cls is None:
                    raise RuntimeError("xFormers lacks LowerTriangularFromBottomRightMask for cached decode")
                attn_bias = bottom_right_cls()
        else:
            visible_rows = None
            attn_bias = None

        out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=float(dropout), scale=scale)
        if visible_rows is not None:
            out = out * visible_rows.transpose(1, 2).unsqueeze(-1).to(out.dtype)
        return out.contiguous(), None

    ALL_MASK_ATTENTION_FUNCTIONS.register("xformers_memory_efficient_aligned", sdpa_mask)
    ALL_ATTENTION_FUNCTIONS.register("xformers_memory_efficient_aligned", xformers_memory_efficient_aligned)
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


def _load_model(model_path: str):
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    start = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=qconfig,
        device_map={"": 0},
        dtype=torch.float16,
        attn_implementation="xformers_memory_efficient_aligned",
    ).eval()
    print(f"loaded_s={time.perf_counter() - start:.3f}", flush=True)
    return tokenizer, model


def _run_single_prompt(tokenizer, model, fla_calls, full_attention_calls, prompt: str, max_new_tokens: int) -> None:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(f"single_input_tokens={inputs['input_ids'].shape[-1]}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        start = time.perf_counter()
        outputs = model(**inputs, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        print(f"MODEL_FORWARD_OK logits_shape={tuple(outputs.logits.shape)} forward_s={time.perf_counter() - start:.3f}", flush=True)
        start = time.perf_counter()
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        print(f"GENERATE_OK tokens={generated.shape[-1]} generate_s={time.perf_counter() - start:.3f}", flush=True)
    print(tokenizer.decode(generated[0], skip_special_tokens=True), flush=True)
    print(f"single_fla_calls={fla_calls}", flush=True)
    print(f"single_full_attention_calls={full_attention_calls}", flush=True)
    print(f"single_max_alloc_gib={torch.cuda.max_memory_allocated() / 1024**3:.3f}", flush=True)


def _run_padded_cache(tokenizer, model, fla_calls, full_attention_calls) -> None:
    for module in model.modules():
        if hasattr(module, "rope_deltas"):
            module.rope_deltas = None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    prompts = ["VLN是什么？", "请用三句话介绍视觉语言导航任务。"]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
    next_ids = tokenizer(" 好", add_special_tokens=False, return_tensors="pt").input_ids[:, :1]
    next_ids = next_ids.to("cuda").expand(inputs["input_ids"].shape[0], -1)
    next_mask = torch.cat(
        [
            inputs["attention_mask"],
            torch.ones(inputs["input_ids"].shape[0], 1, device="cuda", dtype=inputs["attention_mask"].dtype),
        ],
        dim=1,
    )
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        prefill = model(**inputs, use_cache=True, logits_to_keep=1)
        cached = model(
            input_ids=next_ids,
            attention_mask=next_mask,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
    torch.cuda.synchronize()
    print(
        "PADDED_CACHE_OK "
        f"batch={inputs['input_ids'].shape[0]} "
        f"prefill_logits={tuple(prefill.logits.shape)} "
        f"cached_logits={tuple(cached.logits.shape)}",
        flush=True,
    )
    print(f"padded_fla_calls={fla_calls}", flush=True)
    print(f"padded_full_attention_calls={full_attention_calls}", flush=True)
    print(f"padded_max_alloc_gib={torch.cuda.max_memory_allocated() / 1024**3:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen3.5-9B 4-bit with xFormers memory-efficient full attention and FLA linear attention."
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/data/huggingface-z/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    )
    parser.add_argument("--prompt", default="VLN是什么？")
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--skip-padded-cache", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(
        f"device={torch.cuda.get_device_name(0)} capability=sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}",
        flush=True,
    )
    print(f"xformers={xformers.__version__}", flush=True)

    full_attention_calls = register_xformers_memory_efficient_backend()
    tokenizer, model = _load_model(args.model_path)
    fla_calls = count_fla_calls(model)
    _run_single_prompt(tokenizer, model, fla_calls, full_attention_calls, args.prompt, args.max_new_tokens)
    if not args.skip_padded_cache:
        _run_padded_cache(tokenizer, model, fla_calls, full_attention_calls)


if __name__ == "__main__":
    main()
