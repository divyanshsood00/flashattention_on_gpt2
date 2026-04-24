import numpy as np
import pytest
import numba
import os

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps


datatype = np.float32


_BACKENDS = [
    pytest.param(
        minitorch.TensorBackend(CudaKernelOps),
        marks=pytest.mark.skipif(not numba.cuda.is_available(), reason="No GPU"),
    )
]

_HAS_FLASH_KERNEL = os.path.exists("minitorch/cuda_kernels/flash_attention_kernel.so")


def _causal_mask(batch_size, nhead, seq_len, backend):
    mask = -np.finfo(datatype).max * np.triu(
        np.ones((batch_size, nhead, seq_len, seq_len), dtype=datatype),
        1,
    )
    return minitorch.tensor_from_numpy(mask, backend=backend, requires_grad=False)


@pytest.mark.skipif(not _HAS_FLASH_KERNEL, reason="flash_attention_kernel.so not found")
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_flash_attention_forward_matches_reference(backend):
    np.random.seed(7)
    batch_size, nhead, seq_len, head_dim = 2, 4, 16, 32

    q_np = np.random.randn(batch_size, nhead, seq_len, head_dim).astype(datatype)
    k_np = np.random.randn(batch_size, nhead, seq_len, head_dim).astype(datatype)
    v_np = np.random.randn(batch_size, nhead, seq_len, head_dim).astype(datatype)

    q = minitorch.tensor_from_numpy(q_np, backend=backend, requires_grad=True)
    k_t = minitorch.tensor_from_numpy(np.transpose(k_np, (0, 1, 3, 2)).copy(), backend=backend, requires_grad=True)
    v = minitorch.tensor_from_numpy(v_np, backend=backend, requires_grad=True)

    out_flash = q.flash_attention(k_t, v, causal=True)

    mask = _causal_mask(batch_size, nhead, seq_len, backend)
    scores = (q @ k_t) / np.sqrt(head_dim)
    probs = minitorch.nn.softmax(scores + mask, dim=3)
    out_ref = probs @ v

    np.testing.assert_allclose(out_flash.to_numpy(), out_ref.to_numpy(), atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not _HAS_FLASH_KERNEL, reason="flash_attention_kernel.so not found")
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_flash_attention_backward_matches_reference(backend):
    np.random.seed(11)
    batch_size, nhead, seq_len, head_dim = 1, 4, 8, 16

    q_np = np.random.randn(batch_size, nhead, seq_len, head_dim).astype(datatype)
    k_np = np.random.randn(batch_size, nhead, seq_len, head_dim).astype(datatype)
    v_np = np.random.randn(batch_size, nhead, seq_len, head_dim).astype(datatype)

    q_flash = minitorch.tensor_from_numpy(q_np.copy(), backend=backend, requires_grad=True)
    k_t_flash = minitorch.tensor_from_numpy(np.transpose(k_np, (0, 1, 3, 2)).copy(), backend=backend, requires_grad=True)
    v_flash = minitorch.tensor_from_numpy(v_np.copy(), backend=backend, requires_grad=True)

    out_flash = q_flash.flash_attention(k_t_flash, v_flash, causal=True)
    out_flash.sum().backward()

    q_ref = minitorch.tensor_from_numpy(q_np.copy(), backend=backend, requires_grad=True)
    k_t_ref = minitorch.tensor_from_numpy(np.transpose(k_np, (0, 1, 3, 2)).copy(), backend=backend, requires_grad=True)
    v_ref = minitorch.tensor_from_numpy(v_np.copy(), backend=backend, requires_grad=True)

    mask = _causal_mask(batch_size, nhead, seq_len, backend)
    scores = (q_ref @ k_t_ref) / np.sqrt(head_dim)
    probs = minitorch.nn.softmax(scores + mask, dim=3)
    out_ref = probs @ v_ref
    out_ref.sum().backward()

    np.testing.assert_allclose(q_flash.grad.to_numpy(), q_ref.grad.to_numpy(), atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(k_t_flash.grad.to_numpy(), k_t_ref.grad.to_numpy(), atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(v_flash.grad.to_numpy(), v_ref.grad.to_numpy(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not _HAS_FLASH_KERNEL, reason="flash_attention_kernel.so not found")
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_multihead_attention_flash_path_matches_standard(backend):
    np.random.seed(23)
    batch_size, seq_len, n_embd, n_head = 2, 16, 64, 4

    x_np = np.random.randn(batch_size, seq_len, n_embd).astype(datatype)

    flash_layer = minitorch.MultiHeadAttention(
        n_embd=n_embd,
        n_head=n_head,
        causal=True,
        p_dropout=0.0,
        bias=False,
        backend=backend,
        use_fused_kernel=True,
    )
    standard_layer = minitorch.MultiHeadAttention(
        n_embd=n_embd,
        n_head=n_head,
        causal=True,
        p_dropout=0.0,
        bias=False,
        backend=backend,
        use_fused_kernel=False,
    )

    # Keep the two modules numerically identical.
    standard_layer.q_projection.weights.value = minitorch.tensor_from_numpy(
        flash_layer.q_projection.weights.value.to_numpy().copy(), backend=backend, requires_grad=True
    )
    standard_layer.k_projection.weights.value = minitorch.tensor_from_numpy(
        flash_layer.k_projection.weights.value.to_numpy().copy(), backend=backend, requires_grad=True
    )
    standard_layer.v_projection.weights.value = minitorch.tensor_from_numpy(
        flash_layer.v_projection.weights.value.to_numpy().copy(), backend=backend, requires_grad=True
    )
    standard_layer.out_projection.weights.value = minitorch.tensor_from_numpy(
        flash_layer.out_projection.weights.value.to_numpy().copy(), backend=backend, requires_grad=True
    )

    x_flash = minitorch.tensor_from_numpy(x_np.copy(), backend=backend, requires_grad=True)
    x_std = minitorch.tensor_from_numpy(x_np.copy(), backend=backend, requires_grad=True)

    out_flash = flash_layer(x_flash)
    out_std = standard_layer(x_std)

    np.testing.assert_allclose(out_flash.to_numpy(), out_std.to_numpy(), atol=1e-3, rtol=1e-3)

    out_flash.sum().backward()
    out_std.sum().backward()

    np.testing.assert_allclose(x_flash.grad.to_numpy(), x_std.grad.to_numpy(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_multihead_attention_flash_toggle_controls_flash_call(backend, monkeypatch):
    np.random.seed(31)

    batch_size, seq_len, n_embd, n_head = 2, 8, 32, 4
    x_np = np.random.randn(batch_size, seq_len, n_embd).astype(datatype)

    called = {"count": 0}

    def _fake_flash(self, k_t, v, causal=True):
        called["count"] += 1
        raise ValueError("flash path invoked")

    monkeypatch.setattr(minitorch.Tensor, "flash_attention", _fake_flash)

    disabled_layer = minitorch.MultiHeadAttention(
        n_embd=n_embd,
        n_head=n_head,
        causal=True,
        p_dropout=0.0,
        bias=False,
        backend=backend,
        use_fused_kernel=False,
        use_flash_attention=False,
    )

    x_disabled = minitorch.tensor_from_numpy(x_np.copy(), backend=backend, requires_grad=True)
    out_disabled = disabled_layer(x_disabled)
    assert out_disabled.shape == (batch_size, seq_len, n_embd)
    assert called["count"] == 0

    enabled_layer = minitorch.MultiHeadAttention(
        n_embd=n_embd,
        n_head=n_head,
        causal=True,
        p_dropout=0.0,
        bias=False,
        backend=backend,
        use_fused_kernel=False,
        use_flash_attention=True,
    )

    x_enabled = minitorch.tensor_from_numpy(x_np.copy(), backend=backend, requires_grad=True)
    with pytest.raises(ValueError, match="flash path invoked"):
        enabled_layer(x_enabled)
    assert called["count"] > 0
