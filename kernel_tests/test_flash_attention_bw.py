import time
import os
import sys

import numpy as np
import torch

from test_utils import TestDecorator

kt = TestDecorator()

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

backend = minitorch.TensorBackend(CudaKernelOps)
datatype = np.float32

if not os.path.exists("minitorch/cuda_kernels/flash_attention_kernel.so"):
    print("flash_attention_kernel.so not found. Build kernels first. Skipping test.")
    sys.exit(0)


@kt.case(atol=1e-2, rtol=1e-2, ntest=3)
def test_flash_attention_bw():
    batch_size, seq_len = kt.bs_sl()
    nhead = kt.nhead
    head_dim = int(np.random.choice([16, 32]))

    q = kt.rand((batch_size, nhead, seq_len, head_dim))
    k = kt.rand((batch_size, nhead, seq_len, head_dim))
    v = kt.rand((batch_size, nhead, seq_len, head_dim))

    k_t = k.transpose(-1, -2).contiguous()
    causal_mask = -np.finfo(datatype).max * np.triu(
        np.ones((batch_size, nhead, seq_len, seq_len), dtype=datatype), 1
    )

    def custom():
        q_mt = minitorch.tensor_from_numpy(q.detach().cpu().numpy(), backend=backend, requires_grad=True)
        k_t_mt = minitorch.tensor_from_numpy(k_t.detach().cpu().numpy(), backend=backend, requires_grad=True)
        v_mt = minitorch.tensor_from_numpy(v.detach().cpu().numpy(), backend=backend, requires_grad=True)

        start_time = time.time()
        out_mt = q_mt.flash_attention(k_t_mt, v_mt, causal=True)
        out_mt.sum().backward()
        end_time = time.time()

        q_grad = torch.tensor(q_mt.grad.to_numpy(), dtype=torch.float32, device=kt.device)
        k_t_grad = torch.tensor(k_t_mt.grad.to_numpy(), dtype=torch.float32, device=kt.device)
        v_grad = torch.tensor(v_mt.grad.to_numpy(), dtype=torch.float32, device=kt.device)
        return [q_grad, k_t_grad, v_grad], end_time - start_time

    def baseline():
        q_mt = minitorch.tensor_from_numpy(q.detach().cpu().numpy(), backend=backend, requires_grad=True)
        k_t_mt = minitorch.tensor_from_numpy(k_t.detach().cpu().numpy(), backend=backend, requires_grad=True)
        v_mt = minitorch.tensor_from_numpy(v.detach().cpu().numpy(), backend=backend, requires_grad=True)
        mask_mt = minitorch.tensor_from_numpy(causal_mask, backend=backend, requires_grad=False)

        start_time = time.time()
        scores = (q_mt @ k_t_mt) / np.sqrt(head_dim)
        probs = minitorch.nn.softmax(scores + mask_mt, dim=3)
        out_mt = probs @ v_mt
        out_mt.sum().backward()
        end_time = time.time()

        q_grad = torch.tensor(q_mt.grad.to_numpy(), dtype=torch.float32, device=kt.device)
        k_t_grad = torch.tensor(k_t_mt.grad.to_numpy(), dtype=torch.float32, device=kt.device)
        v_grad = torch.tensor(v_mt.grad.to_numpy(), dtype=torch.float32, device=kt.device)
        return [q_grad, k_t_grad, v_grad], end_time - start_time

    return custom, baseline


kt.init(device="cuda:0", nhead=8)
kt.run("test_flash_attention_bw")
