import minitorch

from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.module import Module
from minitorch.modules_basic import Linear as BaseLinear


def _default_backend():
    return minitorch.TensorBackend(CudaKernelOps)


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, backend=None):
        super().__init__()
        if backend is None:
            backend = _default_backend()
        self.layer = BaseLinear(in_size, out_size, bias=True, backend=backend)

    def forward(self, x):
        return self.layer(x)


class Network(Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, backend=None):
        super().__init__()
        if backend is None:
            backend = _default_backend()
        self.backend = backend
        self.l1 = BaseLinear(embedding_dim, hidden_dim, bias=True, backend=backend)
        self.l2 = BaseLinear(hidden_dim, 1, bias=True, backend=backend)

    def forward(self, x):
        # x: (batch, seq_len, embedding_dim)
        batch_size, seq_len, emb_dim = x.shape
        pooled = x.sum(dim=1) / seq_len
        hidden = self.l1(pooled).relu()
        logits = self.l2(hidden)
        probs = logits.sigmoid().view(batch_size)
        return probs
