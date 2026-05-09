# RTX 2080 Ti 上 Triton Flash Attention 可用配置记录

更新时间：2026-04-22

## 结论

在 `NVIDIA GeForce RTX 2080 Ti` 上，Triton 官方教程版 `06-fused-attention.py` 不能直接作为完整的前后向实现使用。

- 官方默认配置：
  - `HEAD_DIM=64` 时，前向可运行。
  - `HEAD_DIM=128` 时，前向会因为 shared memory 超限失败。
  - `HEAD_DIM=64/128` 时，反向都会因为 shared memory 超限失败。
- 经过针对 `sm75` 的 tile 和 stage 缩配后，`causal + fp16` 的前向和反向可以正常运行。

## 测试环境

- GPU：`NVIDIA GeForce RTX 2080 Ti`
- Compute Capability：`sm_75`
- PyTorch：`2.7.1+cu126`
- Triton：`3.5.1`
- 驱动：`580.126.09`
- 测试日期：`2026-04-22`
- 测试卡：`1 号卡`

## 官方默认配置的问题

官方教程实现见 [06-fused-attention.py](/home/z/文档/triton/06-fused-attention.py)。

反向默认配置定义在 [06-fused-attention.py](/home/z/文档/triton/06-fused-attention.py#L575) 和 [06-fused-attention.py](/home/z/文档/triton/06-fused-attention.py#L576)：

- `NUM_WARPS, NUM_STAGES = 4, 5`
- `BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32`

在 2080 Ti 上，实测会触发以下问题：

- 前向，`HEAD_DIM=128`：
  - shared memory 需求 `98304`
  - 硬件上限 `65536`
- 反向，`HEAD_DIM=64/128`：
  - shared memory 需求 `81920`
  - 硬件上限 `65536`

因此，Ampere 之前的 `sm75` 设备不能直接使用教程里的默认 block 配置。

## 已验证可用配置

测试脚本见 [test_flash_attention_2080ti.py](/home/z/文档/triton/test_flash_attention_2080ti.py)。

`sm75` 配置入口定义在 [test_flash_attention_2080ti.py](/home/z/文档/triton/test_flash_attention_2080ti.py#L35) 到 [test_flash_attention_2080ti.py](/home/z/文档/triton/test_flash_attention_2080ti.py#L53)。

### 推荐配置

前向配置：

- `BLOCK_M = 64`
- `BLOCK_N = 32`
- `num_stages = 1`
- `num_warps = 4`

反向配置：

- `BLOCK_M1 = 32`
- `BLOCK_N1 = 32`
- `BLOCK_M2 = 32`
- `BLOCK_N2 = 32`
- `num_stages = 1`
- `num_warps = 4`

这个配置会由脚本生成 [06-fused-attention-sm75.py](/home/z/文档/triton/06-fused-attention-sm75.py) 作为测试用副本。

## 已验证通过的 case

使用命令：

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 python /home/z/文档/triton/test_flash_attention_2080ti.py --profile sm75 --cases 128x64 128x128 1024x64 1024x128
```

通过结果：

- `N_CTX=128, HEAD_DIM=64`
  - `max_abs_out=0.000488`
  - `max_abs_dq=0.001953`
  - `max_abs_dk=0.001221`
  - `max_abs_dv=0.001953`
- `N_CTX=128, HEAD_DIM=128`
  - `max_abs_out=0.000977`
  - `max_abs_dq=0.003906`
  - `max_abs_dk=0.003906`
  - `max_abs_dv=0.003906`
- `N_CTX=1024, HEAD_DIM=64`
  - `max_abs_out=0.000488`
  - `max_abs_dq=0.001953`
  - `max_abs_dk=0.001953`
  - `max_abs_dv=0.001953`
- `N_CTX=1024, HEAD_DIM=128`
  - `max_abs_out=0.000977`
  - `max_abs_dq=0.003906`
  - `max_abs_dk=0.004150`
  - `max_abs_dv=0.003906`

## Qwen3.5-9B / `HEAD_DIM=256` 前向、反向和 KV cache 验证

更新时间：2026-05-09

Qwen3.5-9B 的 full-attention 层使用 `HEAD_DIM=256`。2080 Ti 上原 `sm75`
配置 `BLOCK_M=64, BLOCK_N=32` 会触发 shared memory 超限：

```text
Required: 98304, Hardware limit: 65536
```

实测更小 tile 的结果：

- `BLOCK_M=32, BLOCK_N=32`：`128x256` 通过。
- `BLOCK_M=32, BLOCK_N=16`：`128x256`、`1024x256` 通过；`N_CTX<32` 不适合直接跑。
- `BLOCK_M=16, BLOCK_N=16`：`16x256`、`128x256`、`1024x256` 通过。
- `BLOCK_M=16, BLOCK_N=32`：数值错误。
- `BLOCK_N<16`：Triton `tl.dot` 编译失败，K 维低于最小要求。

推荐的推理 / 训练路径是：

1. 无梯度 full-attention prefill 使用 `BLOCK_M=16, BLOCK_N=16,
   num_stages=1, num_warps=4`，并在 Python backend wrapper 中把 `N_CTX`
   pad 到 16 的倍数后 slice 回原始长度。
2. 需要反向时，unmasked prefill 也走本地 `_CausalAttention` autograd，
   避开教程 backward 在 `HEAD_DIM=256` 上的 shared-memory 超限。
3. masked prefill / cached decode 统一使用本地 causal attention kernels，
   支持 `q_len <= kv_len` 和 `HEAD_DIM=256`。
4. decode 的 causal 位置按 suffix 语义计算：
   `query_abs_pos = kv_len - q_len + query_idx`。

随机张量验证已覆盖：

```text
N_CTX in {1, 4, 5, 8, 15, 16, 20, 31, 32, 128}, HEAD_DIM=256
```

均与 PyTorch causal attention 在 `atol=1e-2, rtol=0` 下对齐。

Qwen3.5-9B 4bit 单卡 GPU3 验证命令：

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 \
  python test_qwen35_9b_triton_sm75.py --generate --max-new-tokens 3
```

实测结果：

- 4bit 加载成功，峰值显存约 `7.479 GiB`。
- full-attention prefill 调用本地 padded Triton backend。
- full-attention cached decode 调用本地 decode kernel。
- linear-attention prefill 调用 FLA `chunk_gated_delta_rule`。
- linear-attention cached decode 调用 FLA `recurrent_gated_delta_rule`。
- `fla_calls={'chunk': 48, 'recurrent': 48}`。
- `full_attention_calls={'n': 32, 'padded': 16, 'decode': 16, 'masked': 0}`。

`use_cache=False` 回归验证：

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 \
  python test_qwen35_9b_triton_sm75.py --generate --max-new-tokens 1 --no-cache
```

结果：

- 峰值显存约 `7.454 GiB`。
- `fla_calls={'chunk': 48, 'recurrent': 0}`。
- `full_attention_calls={'n': 16, 'padded': 16, 'decode': 0, 'masked': 0}`。

额外 KV cache 覆盖：

- 手动 `past_key_values` + 一次送入 `2` 个新 token 通过：
  `full_attention_calls={'n': 16, 'padded': 8, 'decode': 8, 'masked': 0}`。
- batch padding + cache decode 通过：
  `full_attention_calls={'n': 16, 'padded': 0, 'decode': 8, 'masked': 8}`，
  `fla_calls={'chunk': 24, 'recurrent': 24}`，峰值显存约 `7.506 GiB`。
  这里的 `masked=8` 对应 fused Triton masked prefill。

mask 支持说明：

- 自定义 attention backend 已注册 Transformers 的 `sdpa_mask` mask interface。
- 无 padding 的标准 causal mask 会跳过显式 mask，继续走本地 Triton padded
  prefill / decode kernel；需要梯度时改走 `_CausalAttention`。
- masked prefill 收到 4D bool mask 时由 `_causal_attention_fwd_kernel`
  直接执行 block-wise online softmax 和 value 聚合。
- cached decode 收到 4D bool mask 时由 `_decode_attention_kernel` 或
  autograd 路径中的 `_causal_attention_fwd_kernel` 直接应用。
- `full_attention_calls['masked']` 计数的是 fused Triton masked prefill。

反向支持说明：

- `_CausalAttention` 保存 forward 输出和 logsumexp，backward 由 Triton kernel
  重新计算 softmax 概率。
- 反向 kernel：
  - `_attention_bwd_delta_kernel`：计算 `delta = sum(out * d_out)`。
  - `_attention_bwd_dq_kernel`：按 query row 计算 `dQ`。
  - `_attention_bwd_dkdv_kernel`：按 key/value row 计算 `dK/dV`。
- 已验证随机张量 case：
  `q_len/kv_len in {(1,1), (4,4), (17,17), (5,5 masked),
  (1,9), (3,11), (4,13 masked)}`，`HEAD_DIM=256`。
- 对齐 PyTorch reference：forward 最大误差 `<= 4.88e-4`，
  `dQ/dK` 最大误差 `<= 3.1e-5`，`dV` 最大误差 `<= 4.88e-4`。
- backend wrapper 训练态 unmasked prefill `N_CTX=5, HEAD_DIM=256` 已通过，
  不再触发教程 backward 的 shared-memory 超限。

当前限制：

- 当前验证范围是 Qwen3.5-9B text full-attention：
  `causal=True`、`dtype=torch.float16`、`HEAD_DIM=256`。
- `q_len > kv_len` 被视为非法 cache 状态。
- head-dim 分块不能简单切 `HEAD_DIM` 后分别 softmax；正确实现需要两阶段算法：
  先跨 head-dim 分块累计完整 `QK^T` 的 softmax 统计量，再按 value/output
  分块写回。这是更大的 kernel 重写；当前更小 tile + padding + decode kernel
  已解决 `HEAD_DIM=256` 的 Qwen3.5-9B 4bit 前向、反向和 KV cache 路径。

## 如何复现

### 本地推理服务

服务脚本：[serve_qwen35_9b.py](/home/z/文档/triton-flash-attention/serve_qwen35_9b.py)

默认配置：

- 模型：本地 `Qwen3.5-9B` snapshot
- 量化：bitsandbytes 4bit NF4
- GPU：通过 `CUDA_VISIBLE_DEVICES` 指定，推荐 `CUDA_VISIBLE_DEVICES=3`
- full-attention：`xformers_memory_efficient_aligned`
- linear-attention：模型内 FLA
- 视觉：启用，支持图片 URL、`data:image/...;base64,...`、本地路径或原始 base64
- 可用上下文：默认 `prompt_tokens + max_new_tokens <= 80000`，
  可用 `QWEN35_MAX_CONTEXT_TOKENS` 覆盖
- 图片预处理：默认 `QWEN35_IMAGE_MIN_PIXELS=65536`、
  `QWEN35_IMAGE_MAX_PIXELS=1048576`
- 多模态 chat template：默认 `QWEN35_ENABLE_THINKING=0`，直接返回答案
- 端口：`127.0.0.1:8000`

启动：

```bash
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m uvicorn serve_qwen35_9b:app --host 127.0.0.1 --port 8000
```

后台启动示例：

```bash
setsid env CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m uvicorn serve_qwen35_9b:app --host 127.0.0.1 --port 8000 \
  > qwen35_server.log 2>&1 < /dev/null &
echo $! > qwen35_server.pid
```

健康检查：

```bash
curl -fsS http://127.0.0.1:8000/health
```

文本生成：

```bash
curl -fsS http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"VLN是什么？","max_new_tokens":64,"temperature":0}'
```

图片生成：

```bash
curl -fsS http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"描述这张图。","images":["/path/to/image.png"],"max_new_tokens":64,"temperature":0}'
```

OpenAI 风格接口：

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"用一句话解释VLN。"}],"max_tokens":64,"temperature":0}'
```

OpenAI 风格图片输入：

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"/path/to/image.png"}},{"type":"text","text":"这张图里有什么？"}]}],"max_tokens":64,"temperature":0}'
```

当前实测：

- `/health` 正常，加载耗时约 `10.5s`，常驻显存约 `7.35 GiB`。
- `/generate` 正常，`max_new_tokens=6` 时峰值显存约 `7.479 GiB`。
- `/generate` 图片输入正常，`64x64` 红色 PNG 返回 `红色`，
  峰值显存约 `7.481 GiB`。
- `/v1/chat/completions` 的 OpenAI 风格 `image_url` 正常，`64x64` 绿色 PNG
  返回 `绿色`，峰值显存约 `7.489 GiB`。
- 每次请求返回 `full_attention_calls` 和 `fla_calls`，可确认 full-attention
  走 xFormers，linear-attention 走 FLA。
- 服务每次请求前会重置 Qwen3.5 的 `rope_deltas`，避免 generation 状态污染
  后续请求。

长上下文边界：

- Qwen3.5-9B 配置和 tokenizer 的理论上限是 `262144` tokens。
- 在 2080 Ti GPU3 上，服务按可用上下文 `80000` tokens 部署，
  即 `prompt_tokens + max_new_tokens <= 80000`；超过后会在模型生成前
  返回 HTTP 400。
- 需要设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，否则长上下文
  更容易因 CUDA 内存碎片失败。
- 已实测 `81920` 和 `82944` tokens 可完成一次短输出；`83968`、
  `86016`、`90112` tokens 会在 FLA prefill 侧 OOM。线上服务保守取
  `80000` tokens 作为可用上限。

### xFormers memory-efficient attention + FLA

新增脚本 [test_qwen35_9b_xformers_fla.py](/home/z/文档/triton-flash-attention/test_qwen35_9b_xformers_fla.py)
用于验证 full-attention 走 xFormers、linear-attention 继续走 FLA。

关键点：

- full-attention 调用 `xformers.ops.memory_efficient_attention`。
- 无 mask prefill 使用 `LowerTriangularMask`。
- 无 mask cached decode 使用 `LowerTriangularFromBottomRightMask`。
- padding / Tensor mask 使用 aligned additive bias：
  底层分配 `[B, H, Q, ceil(K/8)*8]`，再切片成真实 `[B, H, Q, K]`，
  保证 `attn_bias.stride(-2) % 8 == 0`。
- 对 fully-masked query row 做显式 zero-out，避免 xFormers cutlass kernel
  对全 masked 行返回非零值。

随机张量验证覆盖：

```text
(q_len, kv_len) in {(1,1), (5,5), (5,7), (3,11), (17,17)}, HEAD_DIM=256
```

其中包含非 8 倍数 `kv_len` 和 all-masked query row。对齐 PyTorch reference
的最大误差 `<= 4.88e-4`，且输出全 finite。

整模型验证命令：

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 \
  python test_qwen35_9b_xformers_fla.py --max-new-tokens 3
```

实测结果：

- `xformers=0.0.31.post1`
- 单样本生成通过：
  - `single_fla_calls={'chunk': 48, 'recurrent': 48}`
  - `single_full_attention_calls={'n': 32, 'prefill': 16, 'decode': 16, 'masked': 0, 'aligned_bias': 0}`
  - 峰值显存约 `7.479 GiB`
- batch padding + KV cache 通过：
  - `padded_fla_calls={'chunk': 72, 'recurrent': 72}`
  - `padded_full_attention_calls={'n': 48, 'prefill': 24, 'decode': 24, 'masked': 16, 'aligned_bias': 16}`
  - 峰值显存约 `7.506 GiB`

注意：脚本在单样本 `generate()` 后会重置模型里的 `rope_deltas`，避免 Qwen3.5
多模态 generation 状态污染后续手动 padded-cache smoke test。

### 1. 测官方默认配置

前向：

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 python /home/z/文档/triton/test_flash_attention_2080ti.py --mode fwd
```

前后向：

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 python /home/z/文档/triton/test_flash_attention_2080ti.py --cases 128x64 1024x64
```

### 2. 测 `sm75` 缩配配置

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 python /home/z/文档/triton/test_flash_attention_2080ti.py --profile sm75 --cases 128x64 128x128 1024x64 1024x128
```

## 使用建议

- 如果目标卡是 `RTX 2080 Ti / sm75`，不要直接使用官方教程中的默认反向配置。
- 如果需要在 `sm75` 上稳定运行，优先使用本文档中的 `sm75` 缩配参数。
- 当前验证范围是：
  - `causal=True`
  - `dtype=torch.float16`
  - `HEAD_DIM in {64, 128}`
  - `N_CTX in {128, 1024}`
- Qwen3.5-9B text full-attention 的 `HEAD_DIM=256` 路径已单独验证：
  - prefill：`BLOCK_M=16, BLOCK_N=16` + padding wrapper
  - cached decode：本地 `_decode_attention_kernel`
  - 反向：本地 `_CausalAttention` autograd + Triton backward kernels
  - linear-attention：FLA `chunk` / `recurrent`
- 还没有验证：
  - `non-causal`
  - `fp8`
  - 大于 `256` 的 `HEAD_DIM`
  - 更广泛的 batch/head 组合

## 相关文件

- 官方教程实现：[06-fused-attention.py](/home/z/文档/triton/06-fused-attention.py)
- `sm75` 测试副本：[06-fused-attention-sm75.py](/home/z/文档/triton/06-fused-attention-sm75.py)
- 测试脚本：[test_flash_attention_2080ti.py](/home/z/文档/triton/test_flash_attention_2080ti.py)
- xFormers + FLA 测试脚本：[test_qwen35_9b_xformers_fla.py](/home/z/文档/triton-flash-attention/test_qwen35_9b_xformers_fla.py)
- 推理服务脚本：[serve_qwen35_9b.py](/home/z/文档/triton-flash-attention/serve_qwen35_9b.py)
