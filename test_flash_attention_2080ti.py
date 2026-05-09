import argparse
import importlib.util
import os
import sys
import time
import traceback
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
TUTORIAL_PATH = ROOT / "06-fused-attention.py"

PROFILE_CONFIGS = {
    "sm75": {
        "fwd": {"BLOCK_M": 64, "BLOCK_N": 32, "num_stages": 1, "num_warps": 4},
        "bwd": "BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 32, 32, 32",
        "description": (
            "profile=sm75 fwd(BLOCK_M=64,BLOCK_N=32,num_stages=1,num_warps=4) "
            "bwd(BLOCK_M1=32,BLOCK_N1=32,BLOCK_M2=32,BLOCK_N2=32,num_stages=1,num_warps=4)"
        ),
    },
    "sm75-hd256-32x32": {
        "fwd": {"BLOCK_M": 32, "BLOCK_N": 32, "num_stages": 1, "num_warps": 4},
        "bwd": "BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 32, 32, 32",
        "description": "profile=sm75-hd256-32x32 fwd(BLOCK_M=32,BLOCK_N=32,num_stages=1,num_warps=4)",
    },
    "sm75-hd256-32x16": {
        "fwd": {"BLOCK_M": 32, "BLOCK_N": 16, "num_stages": 1, "num_warps": 4},
        "bwd": "BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 32, 32, 32",
        "description": "profile=sm75-hd256-32x16 fwd(BLOCK_M=32,BLOCK_N=16,num_stages=1,num_warps=4)",
    },
    "sm75-hd256-16x32": {
        "fwd": {"BLOCK_M": 16, "BLOCK_N": 32, "num_stages": 1, "num_warps": 4},
        "bwd": "BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 32, 32, 32",
        "description": "profile=sm75-hd256-16x32 fwd(BLOCK_M=16,BLOCK_N=32,num_stages=1,num_warps=4)",
    },
    "sm75-hd256-16x16": {
        "fwd": {"BLOCK_M": 16, "BLOCK_N": 16, "num_stages": 1, "num_warps": 4},
        "bwd": "BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 32, 32, 32",
        "description": "profile=sm75-hd256-16x16 fwd(BLOCK_M=16,BLOCK_N=16,num_stages=1,num_warps=4)",
    },
    "sm75-hd256-8x16": {
        "fwd": {"BLOCK_M": 8, "BLOCK_N": 16, "num_stages": 1, "num_warps": 4},
        "bwd": "BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 32, 32, 32",
        "description": "profile=sm75-hd256-8x16 fwd(BLOCK_M=8,BLOCK_N=16,num_stages=1,num_warps=4)",
    },
    "sm75-hd256-4x16": {
        "fwd": {"BLOCK_M": 4, "BLOCK_N": 16, "num_stages": 1, "num_warps": 4},
        "bwd": "BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 32, 32, 32",
        "description": "profile=sm75-hd256-4x16 fwd(BLOCK_M=4,BLOCK_N=16,num_stages=1,num_warps=4)",
    },
    "sm75-hd256-8x8": {
        "fwd": {"BLOCK_M": 8, "BLOCK_N": 8, "num_stages": 1, "num_warps": 4},
        "bwd": "BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 32, 32, 32",
        "description": "profile=sm75-hd256-8x8 fwd(BLOCK_M=8,BLOCK_N=8,num_stages=1,num_warps=4)",
    },
    "sm75-hd256-4x4": {
        "fwd": {"BLOCK_M": 4, "BLOCK_N": 4, "num_stages": 1, "num_warps": 4},
        "bwd": "BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 32, 32, 32",
        "description": "profile=sm75-hd256-4x4 fwd(BLOCK_M=4,BLOCK_N=4,num_stages=1,num_warps=4)",
    },
    "sm75-hd256-1x1": {
        "fwd": {"BLOCK_M": 1, "BLOCK_N": 1, "num_stages": 1, "num_warps": 4},
        "bwd": "BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 32, 32, 32",
        "description": "profile=sm75-hd256-1x1 fwd(BLOCK_M=1,BLOCK_N=1,num_stages=1,num_warps=4)",
    },
}


class _FakeMark:
    def parametrize(self, *args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


def _load_tutorial_module(use_single_config, profile):
    if use_single_config:
        os.environ.setdefault("PYTEST_VERSION", "1")
    else:
        os.environ.pop("PYTEST_VERSION", None)
    fake_pytest = types.SimpleNamespace(
        mark=_FakeMark(),
        skip=lambda msg="": (_ for _ in ()).throw(RuntimeError(f"pytest.skip called: {msg}")),
    )
    sys.modules.setdefault("pytest", fake_pytest)
    module_path = TUTORIAL_PATH
    if profile in PROFILE_CONFIGS:
        profile_config = PROFILE_CONFIGS[profile]
        fwd = profile_config["fwd"]
        text = TUTORIAL_PATH.read_text()
        prefix, rest = text.split("configs = [", 1)
        _, suffix = rest.split("def keep", 1)
        custom_configs = (
            "configs = [\n"
            f"    triton.Config({{'BLOCK_M': {fwd['BLOCK_M']}, 'BLOCK_N': {fwd['BLOCK_N']}}}, "
            f"num_stages={fwd['num_stages']}, num_warps={fwd['num_warps']}, "
            "pre_hook=_host_descriptor_pre_hook),\n"
            "]\n\n\n"
            "def keep"
        )
        text = prefix + custom_configs + suffix
        text = text.replace("NUM_WARPS, NUM_STAGES = 4, 5", "NUM_WARPS, NUM_STAGES = 4, 1")
        text = text.replace(
            "BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32",
            profile_config["bwd"],
        )
        module_path = ROOT / ("06-fused-attention-sm75.py" if profile == "sm75" else f"06-fused-attention-{profile}.py")
        module_path.write_text(text)
    spec = importlib.util.spec_from_file_location("triton_flash_tutorial", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_inputs(batch, heads, n_ctx, head_dim, dtype, device):
    q = torch.empty((batch, heads, n_ctx, head_dim), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    k = torch.empty((batch, heads, n_ctx, head_dim), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    v = torch.empty((batch, heads, n_ctx, head_dim), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    return q, k, v


def _run_case(module, batch, heads, n_ctx, head_dim, causal, sm_scale, dtype, mode):
    device = torch.device("cuda")
    torch.manual_seed(20)

    q_ref, k_ref, v_ref = _make_inputs(batch, heads, n_ctx, head_dim, dtype, device)
    q_tri = q_ref.clone().detach().requires_grad_(True)
    k_tri = k_ref.clone().detach().requires_grad_(True)
    v_tri = v_ref.clone().detach().requires_grad_(True)
    q_ref = q_ref.clone().detach().requires_grad_(True)
    k_ref = k_ref.clone().detach().requires_grad_(True)
    v_ref = v_ref.clone().detach().requires_grad_(True)

    mask = torch.tril(torch.ones((n_ctx, n_ctx), device=device, dtype=torch.bool))
    scores = torch.matmul(q_ref, k_ref.transpose(2, 3)) * sm_scale
    if causal:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(dtype)
    ref_out = torch.matmul(probs, v_ref).half()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tri_out = module.attention(q_tri, k_tri, v_tri, causal, sm_scale, False).half()
    torch.cuda.synchronize()
    fwd_elapsed_s = time.perf_counter() - t0

    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0.0)

    result = {
        "fwd_elapsed_s": fwd_elapsed_s,
        "max_abs_out": (tri_out - ref_out).abs().max().item(),
    }

    if mode == "fwd":
        return result

    dout = torch.randn_like(ref_out)
    ref_out.backward(dout)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tri_out.backward(dout)
    torch.cuda.synchronize()
    bwd_elapsed_s = time.perf_counter() - t0

    torch.testing.assert_close(q_tri.grad, q_ref.grad, atol=1e-2, rtol=0.0)
    torch.testing.assert_close(k_tri.grad, k_ref.grad, atol=1e-2, rtol=0.0)
    torch.testing.assert_close(v_tri.grad, v_ref.grad, atol=1e-2, rtol=0.0)

    result.update(
        {
            "bwd_elapsed_s": bwd_elapsed_s,
            "max_abs_dq": (q_tri.grad - q_ref.grad).abs().max().item(),
            "max_abs_dk": (k_tri.grad - k_ref.grad).abs().max().item(),
            "max_abs_dv": (v_tri.grad - v_ref.grad).abs().max().item(),
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Test Triton fused attention on RTX 2080 Ti.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--mode", choices=["fwd", "both"], default="both")
    parser.add_argument("--profile", choices=["official", *PROFILE_CONFIGS], default="official")
    parser.add_argument(
        "--full-autotune",
        action="store_true",
        help="Use the tutorial's full autotune config set instead of a single reduced config.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=["128x64", "128x128", "1024x64"],
        help="Sequence length and head dim pairs in the form N_CTXxHEAD_DIM.",
    )
    parser.add_argument("--sm-scale", type=float, default=0.5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    cap = torch.cuda.get_device_capability()
    print(f"device={torch.cuda.get_device_name(0)} capability=sm_{cap[0]}{cap[1]}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    try:
        import triton
        print(f"triton={triton.__version__}", flush=True)
    except Exception as exc:
        print(f"triton import failed: {exc}", flush=True)

    if args.profile in PROFILE_CONFIGS:
        print(PROFILE_CONFIGS[args.profile]["description"], flush=True)
    module = _load_tutorial_module(use_single_config=not args.full_autotune, profile=args.profile)
    dtype = torch.float16

    failed = False
    for case in args.cases:
        n_ctx_text, head_dim_text = case.lower().split("x", 1)
        n_ctx = int(n_ctx_text)
        head_dim = int(head_dim_text)
        print(f"RUN n_ctx={n_ctx} head_dim={head_dim} mode={args.mode}", flush=True)
        try:
            result = _run_case(
                module=module,
                batch=args.batch,
                heads=args.heads,
                n_ctx=n_ctx,
                head_dim=head_dim,
                causal=True,
                sm_scale=args.sm_scale,
                dtype=dtype,
                mode=args.mode,
            )
            if args.mode == "fwd":
                print(
                    "PASS "
                    f"fwd={result['fwd_elapsed_s']:.3f}s "
                    f"max_abs_out={result['max_abs_out']:.6f}",
                    flush=True,
                )
            else:
                print(
                    "PASS "
                    f"fwd={result['fwd_elapsed_s']:.3f}s "
                    f"bwd={result['bwd_elapsed_s']:.3f}s "
                    f"max_abs_out={result['max_abs_out']:.6f} "
                    f"max_abs_dq={result['max_abs_dq']:.6f} "
                    f"max_abs_dk={result['max_abs_dk']:.6f} "
                    f"max_abs_dv={result['max_abs_dv']:.6f}",
                    flush=True,
                )
        except Exception:
            failed = True
            print(f"FAIL n_ctx={n_ctx} head_dim={head_dim}", flush=True)
            traceback.print_exc()

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
