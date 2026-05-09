import argparse
import functools
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv

from test_flash_attention_2080ti import _load_tutorial_module


def register_triton_sm75_hd256_padded_backend() -> dict[str, int]:
    triton_mod = _load_tutorial_module(use_single_config=True, profile="sm75-hd256-16x16")
    calls = {"n": 0, "padded": 0}

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
        if attention_mask is not None:
            raise RuntimeError("attention_mask is not implemented in the local Triton backend")
        if hasattr(module, "num_key_value_groups") and module.num_key_value_groups != 1:
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        if query.shape[2] != key.shape[2] or key.shape[2] != value.shape[2]:
            raise RuntimeError(
                "local Triton backend requires q/k/v to have the same sequence length, "
                f"got {query.shape[2]}/{key.shape[2]}/{value.shape[2]}"
            )

        n_ctx = query.shape[2]
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
    parser.add_argument("--generate", action="store_true", help="Run one-token generate(use_cache=False) after loading.")
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
            generated = model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=False)
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
