import torch

import triton
import triton.language as tl
from triton.language.math import (
    fast_dividef,
    fast_expf,
)


def naive_torch_gated_ffn_pt1(x, w_g, w_fc):
    gate = torch.nn.SiLU(torch.matmul(x, w_g))
    fc = torch.matmul(x, w_fc)
    y = gate * fc
    return y


@triton.jit
def fast_silu(x):
    return fast_dividef(x, 1.0 + fast_expf(-x))


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
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
    y = torch.empty((B, M, N), device=a.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B)
    fused_gated_ffn_pt1_kernel[grid](
        x, w_g, w_fc, y,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        w_g.stride(0), w_g.stride(1),
        y.stride(0), y.stride(1), y.stride(2),
    )
    return y


# %%
# Unit Test
# ---------
#

torch.manual_seed(0)
x = torch.randn((32, 256, 256), device='cuda', dtype=torch.float16)
w_g = torch.randn((256, 256), device='cuda', dtype=torch.float16)
w_fc = torch.randn((256, 256), device='cuda', dtype=torch.float16)
triton_output = fused_gated_ffn_pt1(x, w_g, w_fc)
torch_output = naive_torch_gated_ffn_pt1(x, w_g, w_fc)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
rtol = 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

# TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
# if TORCH_HAS_FP8 and is_cuda():
#     torch.manual_seed(0)
#     a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
#     b = torch.randn((512, 512), device="cuda", dtype=torch.float16)
#     a = a.to(torch.float8_e5m2)
#     # pre-transpose b for efficiency.
#     b = b.T
#     b = b.to(torch.float8_e5m2)
#     triton_output = matmul(a, b)
#     torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
#     print(f"triton_output_with_fp8_inputs={triton_output}")
#     print(f"torch_output_with_fp8_inputs={torch_output}")
#     if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
#         print("✅ Triton and Torch match")
#     else:
#         print("❌ Triton and Torch differ")

# # %%
# # Benchmark
# # ---------
# #
# # Square Matrix Performance
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~
# #
# # We can now compare the performance of our kernel against that of cuBLAS or rocBLAS. Here we focus on square matrices,
# # but feel free to arrange this script as you wish to benchmark any other matrix shape.

# ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

# configs = []
# for fp8_inputs in [False, True]:
#     if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
#         continue
#     configs.append(
#         triton.testing.Benchmark(
#             x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
#             x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
#             line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
#             # Possible values for `line_arg`
#             # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
#             line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
#             line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
#             styles=[("green", "-"), ("blue", "-")],
#             ylabel="TFLOPS",  # Label name for the y-axis
#             plot_name="matmul-performance-" +
#             ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
#             args={"fp8_inputs": fp8_inputs},
#         ))


# @triton.testing.perf_report(configs)
# def benchmark(M, N, K, provider, fp8_inputs):
#     a = torch.randn((M, K), device='cuda', dtype=torch.float16)
#     b = torch.randn((K, N), device='cuda', dtype=torch.float16)
#     if TORCH_HAS_FP8 and fp8_inputs:
#         a = a.to(torch.float8_e5m2)
#         b = b.T
#         b = b.to(torch.float8_e5m2)
#     quantiles = [0.5, 0.2, 0.8]
#     if provider == ref_lib.lower():
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
#     if provider == 'triton':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
#     perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
#     return perf(ms), perf(max_ms), perf(min_ms)


# benchmark.run(show_plots=True, print_data=True)
