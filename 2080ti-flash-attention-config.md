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

## Qwen3.5-9B / `HEAD_DIM=256` 前向验证

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

推荐的推理前向路径是：

1. full-attention 使用 `BLOCK_M=16, BLOCK_N=16, num_stages=1, num_warps=4`。
2. 在 Python backend wrapper 中把 `N_CTX` pad 到 16 的倍数。
3. 调用 Triton attention 后 slice 回原始 `N_CTX`。

随机张量验证已覆盖：

```text
N_CTX in {1, 4, 5, 8, 15, 16, 20, 31, 32, 128}, HEAD_DIM=256
```

均与 PyTorch causal attention 在 `atol=1e-2, rtol=0` 下对齐。

Qwen3.5-9B 4bit 单卡 GPU3 验证命令：

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 \
  python test_qwen35_9b_triton_sm75.py --generate
```

实测结果：

- 4bit 加载成功，峰值显存约 `7.454 GiB`。
- full-attention 调用本地 padded Triton backend。
- linear-attention 调用 FLA `chunk_gated_delta_rule`。
- `generate(..., max_new_tokens=1, use_cache=False)` 成功。

当前限制：

- 本地 Triton full-attention backend 只覆盖 prefill / `use_cache=False`，要求
  `q/k/v` 序列长度相同。
- cached decode 的 full-attention 层会出现 `q_len != kv_len`，还需要单独实现
  decode kernel 或在该路径 fallback。
- head-dim 分块不能简单切 `HEAD_DIM` 后分别 softmax；正确实现需要两阶段算法：
  先跨 head-dim 分块累计完整 `QK^T` 的 softmax 统计量，再按 value/output
  分块写回。这是更大的 kernel 重写；当前更小 tile + padding 已解决
  `HEAD_DIM=256` prefill 前向。

## 如何复现

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
- 还没有验证：
  - `non-causal`
  - `fp8`
  - 更大的 `HEAD_DIM`
  - 更广泛的 batch/head 组合

## 相关文件

- 官方教程实现：[06-fused-attention.py](/home/z/文档/triton/06-fused-attention.py)
- `sm75` 测试副本：[06-fused-attention-sm75.py](/home/z/文档/triton/06-fused-attention-sm75.py)
- 测试脚本：[test_flash_attention_2080ti.py](/home/z/文档/triton/test_flash_attention_2080ti.py)
