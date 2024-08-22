import torch

import triton
import triton.language as tl
from triton.language.math import (
    fast_dividef,
    fast_expf,
)


def naive_torch_gated_ffn_pt1(x, w_g, w_fc):
    gate = torch.nn.functional.silu(torch.matmul(x, w_g))
    fc = torch.matmul(x, w_fc)
    y = gate * fc
    return y


@triton.jit
def fast_silu(x):
    return fast_dividef(x, 1.0 + fast_expf(-x))


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_autotune_config():
    configs = [
        triton.Config({"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GM}, num_stages=s, num_warps=w)  \
        for BM in [64, 128]  \
        for BN in [64, 128]  \
        for BK in [32]  \
        for GM in [8]  \
        for s in [2, 4]  \
        for w in [4, 8]
    ]
    return configs


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gated_ffn_pt1_kernel(
        x_ptr, w_g_ptr, w_fc_ptr, y_ptr,
        M, N, K,
        stride_xb, stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_yb, stride_ym, stride_yn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offset_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offset_b * stride_xb + offset_xm[:, None] * stride_xm + offset_k[None, :] * stride_xk)
    w_g_ptrs = w_g_ptr + (offset_k[:, None] * stride_wk + offset_wn[None, :] * stride_wn)
    w_fc_ptrs = w_fc_ptr + (offset_k[:, None] * stride_wk + offset_wn[None, :] * stride_wn)

    accumulator_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator_fc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offset_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w_g = tl.load(w_g_ptrs, mask=offset_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        w_fc = tl.load(w_fc_ptrs, mask=offset_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator_g = tl.dot(x, w_g, accumulator_g)
        accumulator_fc = tl.dot(x, w_fc, accumulator_fc)
        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_g_ptrs += BLOCK_SIZE_K * stride_wk
        w_fc_ptrs += BLOCK_SIZE_K * stride_wk
    accumulator_g = fast_silu(accumulator_g)
    hadamard_product = accumulator_g * accumulator_fc
    y = hadamard_product.to(tl.float16)

    offset_ym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = y_ptr + offset_b * stride_yb + stride_ym * offset_ym[:, None] + stride_yn * offset_yn[None, :]
    y_mask = (offset_ym[:, None] < M) & (offset_yn[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)


def fused_gated_ffn_pt1(x, w_g, w_fc):
    # Check constraints.
    assert w_g.shape == w_fc.shape
    assert x.shape[2] == w_g.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "Tensor X must be contiguous"
    assert x.dtype == w_g.dtype == w_fc.dtype and x.dtype in [torch.bfloat16, torch.float16]
    B, M, K = x.shape
    K, N = w_g.shape
    # Allocates output.
    y = torch.empty((B, M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B)
    fused_gated_ffn_pt1_kernel[grid](
        x, w_g, w_fc, y,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        w_g.stride(0), w_g.stride(1),
        y.stride(0), y.stride(1), y.stride(2),
    )
    #print(fused_gated_ffn_pt1_kernel.best_config)
    return y


# %%
# Unit Test
# ---------
#

torch.manual_seed(0)
x = torch.randn((128, 256, 2048), device='cuda', dtype=torch.float16)
w_g = torch.randn((2048, 4096), device='cuda', dtype=torch.float16)
w_fc = torch.randn((2048, 4096), device='cuda', dtype=torch.float16)
triton_output = fused_gated_ffn_pt1(x, w_g, w_fc)
torch_output = naive_torch_gated_ffn_pt1(x, w_g, w_fc)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
rtol = 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

# %%
# Benchmark
# ---------
#

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[(256, 4096, 2048), (384, 4096, 2048), (512, 4096, 2048),
                (640, 4096, 2048), (2048, 512, 256), (2048, 1532, 768),
                (4096, 512, 256), (4096, 1532, 768), (5120, 512, 256),
                (5120, 1532, 768)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Torch", "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name="gated-ffn-pt1-performance-fp16",
        args={}
    ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    x = torch.randn((128, M, K), device='cuda', dtype=torch.float16)
    w_g = torch.randn((K, N), device='cuda', dtype=torch.float16)
    w_fc = torch.randn((K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_torch_gated_ffn_pt1(x, w_g, w_fc), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_gated_ffn_pt1(x, w_g, w_fc), quantiles=quantiles)
    return ms, max_ms, min_ms


benchmark.run(show_plots=True, print_data=True)
