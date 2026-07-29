# Issue draft

## Title

[Inductor][Triton][CUDA][sm_120] BF16 autocast fusion of embeddings, Linear, and RMSNorm silently produces wrong results

## Description

`torch.compile` silently produces materially incorrect CUDA results when a
BF16-autocast `Linear` output is added to FP32 embedding outputs and the result
feeds an RMSNorm-style reduction. The eager result is finite and stable. The
compiled result is also finite, but its maximum absolute error is approximately
`2.2` in the small reproducer below.

This is a standard autocast pattern. Model parameters are FP32. Embedding
outputs remain FP32, the autocast `Linear` output is BF16, and the addition is
promoted to FP32 before the FP32 RMSNorm reduction. The model is in evaluation
and inference modes. Inputs are contiguous and are not mutated.

The issue reproduces on an RTX 5080 (`sm_120`). It is shape-dependent. The
provided `batch=96`, `length=8`, `width=192` shape reproduces consistently.

## Reproduction

```python
import platform

import torch
from torch import nn

BATCH = 96
LENGTH = 8
WIDTH = 192


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(4, WIDTH) for _ in range(5))
        self.numeric = nn.Linear(2, WIDTH, bias=False)
        self.weight = nn.Parameter(torch.ones(WIDTH))

    def forward(self, indices, values):
        hidden = self.embeddings[0](indices[:, :, 0])
        for column, embedding in enumerate(self.embeddings[1:], start=1):
            hidden = hidden + embedding(indices[:, :, column])
        hidden = hidden + self.numeric(values)
        scale = hidden.float().square().mean(dim=-1, keepdim=True)
        return hidden * torch.rsqrt(scale + 1e-6).to(hidden.dtype) * self.weight


torch.manual_seed(0)
device = torch.device("cuda")
model = Model().to(device).eval()
indices = torch.arange(BATCH * LENGTH * 5, device=device).reshape(
    BATCH, LENGTH, 5
) % 4
values = torch.arange(
    BATCH * LENGTH * 2, device=device, dtype=torch.float32
).reshape(BATCH, LENGTH, 2) / 100.0
compiled = torch.compile(model, fullgraph=True, dynamic=False)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    expected = model(indices, values)
    actual = compiled(indices, values)
torch.cuda.synchronize()

delta = (expected.float() - actual.float()).abs()
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__} ({torch.version.git_version})")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"different: {(expected != actual).sum().item()}")
print(f"nonfinite: {(~torch.isfinite(actual)).sum().item()}")
print(f"max_delta: {delta.max().item():.9g}")
```

Actual output:

```text
Python: 3.14.6
PyTorch: 2.13.0+cu130 (cf30153c4c131c8164ee7798e5022d810682e2cb)
CUDA: 13.0
GPU: NVIDIA GeForce RTX 5080
different: 37901
nonfinite: 0
max_delta: 2.20142961
```

## Expected behavior

The compiled result can differ by normal BF16 rounding, but it must not have
an absolute error of `2.2`. For comparison, removing the autocast `Linear`
branch gives a maximum error of `2.38e-7`. Converting all model weights to
native BF16 before compiling gives a maximum error of `0.015625`.

In the original Transformer policy, the default Inductor mode changes 2 to 3
of 128 greedy actions and has a maximum raw-logit error between `16` and `20`.
`reduce-overhead` has also produced hundreds of non-finite logits.

## Triage results

- Dynamo eager, AOT eager, the cudagraph backend, `torch.export`, and
  TorchScript trace do not reproduce the large error.
- FP32 Inductor has a maximum error below `3.3e-4` on the full history encoder.
- Native BF16 weights avoid the large error.
- Forcing ATen-only or Triton-only GEMM does not remove the error.
- Disabling epilogue fusion, layout optimization, pattern matching, or in-place
  buffers does not remove the error.
- Setting `torch._inductor.config.max_fusion_size = 1` reduces the minimal
  padded reproducer from a maximum error of `56.6` to `2.38e-7`.
- Detailed logging reports no graph breaks or recompilations.

The generated faulty kernel is named similar to:

```text
triton_red_fused__to_copy__unsafe_view_add_clamp_embedding_lt_mean_mul_pow_rsqrt_scalar_tensor_select_where_2
```

Its metadata contains `mutated_arg_names=['in_out_ptr0']`. The kernel fuses
embedding lookup and addition, the RMS reduction, `rsqrt`, and output
multiplication. It first stores an FP32 embedding intermediate into
`in_out_ptr0`, then reads that buffer in a second reduction loop and adds the
BF16 Linear result. Preventing that fusion restores correctness.

Logs can be collected with:

```bash
TORCH_LOGS=output_code,kernel_code,graph_breaks,recompiles,fusion \
TORCH_COMPILE_DEBUG=1 \
python torch_compile_issue_repro.py
```

## Environment

```text
PyTorch version: 2.13.0+cu130
PyTorch commit: cf30153c4c131c8164ee7798e5022d810682e2cb
Triton version: 3.7.1
CUDA used to build PyTorch: 13.0
CUDA runtime version: 13.2.86
NVIDIA driver: 595.80
GPU: NVIDIA GeForce RTX 5080
OS: Fedora Linux 44, Linux 7.1.3-201.fc44, x86_64
Python: 3.14.6, Anaconda, GCC 14.3.0
cuDNN package: 9.20.0.48
```
