"""Small standalone reproducer for a TorchInductor reduction-fusion error."""

import platform

import torch
from torch import nn


BATCH = 96
LENGTH = 8
WIDTH = 192


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(4, WIDTH) for _ in range(5))
        self.numeric = nn.Linear(2, WIDTH, bias=False)
        self.weight = nn.Parameter(torch.ones(WIDTH))

    def forward(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        hidden = self.embeddings[0](indices[:, :, 0])
        for column, embedding in enumerate(self.embeddings[1:], start=1):
            hidden = hidden + embedding(indices[:, :, column])
        hidden = hidden + self.numeric(values)
        scale = hidden.float().square().mean(dim=-1, keepdim=True)
        return hidden * torch.rsqrt(scale + 1e-6).to(hidden.dtype) * self.weight


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
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


if __name__ == "__main__":
    main()
