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
