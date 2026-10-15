"""Minimal example: a transformer MLP block through tensorjit and through torch.compile."""
import torch
import torch.nn.functional as F

import tensorjit as tj


@tj.jit
def mlp_block(x, ln_g, w1, b1, w2, b2):
    h = tj.layer_norm(x, ln_g)
    return x + tj.linear(tj.gelu(tj.linear(h, w1, b1)), w2, b2)


if __name__ == "__main__":
    T, D = 512, 256
    x, g = torch.randn(T, D), torch.ones(D)
    w1, b1, w2, b2 = torch.randn(4 * D, D), torch.randn(4 * D), torch.randn(D, 4 * D), torch.randn(D)
    out = mlp_block(x, g, w1, b1, w2, b2)
    ref = x + F.linear(F.gelu(F.linear(F.layer_norm(x, (D,), g), w1, b1)), w2, b2)
    print("max abs err:", (out - ref).abs().max().item())
    print(mlp_block.program(x, g, w1, b1, w2, b2))           # fused kernel plan
    print(mlp_block.compiled(x, g, w1, b1, w2, b2).source[:800])  # generated code

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ln, self.fc1, self.fc2 = torch.nn.LayerNorm(D), torch.nn.Linear(D, 4 * D), torch.nn.Linear(4 * D, D)

        def forward(self, x):
            return x + self.fc2(F.gelu(self.fc1(self.ln(x))))

    m = Block().eval()
    compiled = torch.compile(m, backend=tj.torch_compile_backend)
    with torch.no_grad():
        print("torch.compile backend err:", (compiled(x) - m(x)).abs().max().item())
