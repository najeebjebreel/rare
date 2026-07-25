"""
GPU smoke-test for the RARE project.

Verifies that the active Python environment can fully exploit the host GPU,
and surfaces version mismatches that would silently degrade performance.

Designed for Blackwell-class GPUs (sm_120: RTX PRO 6000 Blackwell, RTX 50xx)
but works on any modern NVIDIA card.

Usage:
    python scripts/check_gpu.py            # capability checks only (~2 s)
    python scripts/check_gpu.py --bench    # also run matmul + WRN-28-2 timing
"""

import _bootstrap  # noqa: F401

import argparse
import os
import platform
import subprocess
import sys
import textwrap
import time


# ---------- pretty printing ----------
class _Tag:
    PASS = "[ OK ]"
    WARN = "[WARN]"
    FAIL = "[FAIL]"
    INFO = "[ -- ]"


def line(tag: str, msg: str) -> None:
    print(f"  {tag}  {msg}")


def header(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------- checks ----------
def check_python() -> None:
    header("Python / OS")
    line(_Tag.INFO, f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()}")
    line(_Tag.INFO, f"Conda env: {os.environ.get('CONDA_DEFAULT_ENV', '<none>')}")
    in_wsl = "microsoft" in platform.uname().release.lower()
    line(_Tag.INFO, f"WSL: {'yes' if in_wsl else 'no'}")


def check_nvidia_smi() -> None:
    header("nvidia-smi (Windows-driver passthrough)")
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            timeout=10,
        ).decode()
        for row in out.strip().splitlines():
            line(_Tag.PASS, row.strip())
    except FileNotFoundError:
        line(_Tag.FAIL, "nvidia-smi not found — Windows driver passthrough not visible inside WSL")
    except Exception as e:
        line(_Tag.FAIL, f"nvidia-smi failed: {e}")


def check_torch():
    header("PyTorch")
    info = {"devices": [], "arch_list": []}
    try:
        import torch
    except ImportError as e:
        line(_Tag.FAIL, f"torch import failed: {e}")
        line(_Tag.INFO, "Install torch first — see WSL_SETUP.md")
        return None, info

    info["torch"] = torch.__version__
    info["cuda_runtime"] = torch.version.cuda or "<cpu-only build>"
    info["cudnn"] = torch.backends.cudnn.version()
    line(_Tag.INFO, f"torch {info['torch']}  |  cuda runtime {info['cuda_runtime']}  |  cuDNN {info['cudnn']}")

    if not torch.cuda.is_available():
        line(_Tag.FAIL, "torch.cuda.is_available() is False — CPU-only build or driver/runtime mismatch")
        return torch, info

    n = torch.cuda.device_count()
    line(_Tag.PASS, f"CUDA available, {n} device(s)")

    arch_list = torch.cuda.get_arch_list()
    info["arch_list"] = arch_list
    line(_Tag.INFO, f"torch built for: {' '.join(arch_list) or '<empty>'}")

    for i in range(n):
        cap = torch.cuda.get_device_capability(i)  # (major, minor)
        sm = f"sm_{cap[0]}{cap[1]}"
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        info["devices"].append({"idx": i, "name": name, "sm": sm, "cap": cap, "mem_gb": mem})
        line(_Tag.INFO, f"cuda:{i}  {name}  {sm}  {mem:.1f} GB")

    have_arch = {a for a in arch_list if a.startswith("sm_")}
    bad = [d for d in info["devices"] if d["sm"] not in have_arch]
    if bad:
        for d in bad:
            line(
                _Tag.WARN,
                f"cuda:{d['idx']} is {d['sm']} but torch was not built for that arch. "
                "Kernels may JIT from PTX (slow first run) or fall back to a slower path."
            )
    else:
        line(_Tag.PASS, "every device's compute capability is in torch's arch list")

    return torch, info


def check_simple_forward(torch) -> bool:
    header("Smoke: matmul + backward")
    try:
        x = torch.randn(1024, 512, device="cuda")
        w = torch.randn(512, 256, device="cuda", requires_grad=True)
        y = x @ w
        y.sum().backward()
        torch.cuda.synchronize()
        line(_Tag.PASS, "matmul + backward executed without error")
        return True
    except Exception as e:
        line(_Tag.FAIL, f"matmul failed: {e}")
        return False


def check_memory_alloc(torch) -> None:
    header("Memory allocation")
    free, total = torch.cuda.mem_get_info()
    line(_Tag.INFO, f"free {free / 1024 ** 3:.1f} GB / total {total / 1024 ** 3:.1f} GB")

    target_gb = max(1, int((total // 1024 ** 3) * 0.5))
    try:
        x = torch.empty(target_gb * 1024 ** 3 // 4, dtype=torch.float32, device="cuda")
        x.zero_()
        torch.cuda.synchronize()
        del x
        torch.cuda.empty_cache()
        line(_Tag.PASS, f"allocated and zeroed {target_gb} GB")
    except Exception as e:
        line(_Tag.WARN, f"could not allocate {target_gb} GB: {e}")


# ---------- benchmarks ----------
def _time_matmul(torch, N: int, dtype, allow_tf32: bool):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    a = torch.randn(N, N, device="cuda", dtype=dtype)
    b = torch.randn(N, N, device="cuda", dtype=dtype)

    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize()

    iters = 10
    t0 = time.time()
    for _ in range(iters):
        _ = a @ b
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    tflops = 2 * (N ** 3) / dt / 1e12
    return dt, tflops


def bench_matmul(torch, sizes=(4096, 8192)):
    header("Benchmark: matmul throughput")
    line(_Tag.INFO, f"{'N':>6} {'dtype':>6} {'TFLOP/s':>10} {'ms/iter':>10}")
    for N in sizes:
        for dtype, label, tf32 in [
            (torch.float32, "fp32",  False),
            (torch.float32, "tf32",  True),
            (torch.bfloat16, "bf16", False),
            (torch.float16,  "fp16", False),
        ]:
            try:
                dt, tflops = _time_matmul(torch, N, dtype, tf32)
                line(_Tag.INFO, f"{N:>6} {label:>6} {tflops:>10.1f} {dt * 1000:>10.2f}")
            except Exception as e:
                line(_Tag.WARN, f"{N:>6} {label:>6}  failed: {e}")


def bench_wrn(torch):
    header("Benchmark: WRN-28-2 forward (CIFAR-shaped input)")
    try:
        from rare.utils.model import get_model
    except Exception as e:
        line(_Tag.WARN, f"could not import rare.utils.model: {e}")
        return

    model = get_model(num_classes=10, architecture="wrn28-2", input_channels=3).cuda().eval()
    x_fp32 = torch.randn(256, 3, 32, 32, device="cuda")

    def _time(model, x, iters=50):
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters):
                _ = model(x)
            torch.cuda.synchronize()
        return (time.time() - t0) / iters

    dt = _time(model, x_fp32)
    line(_Tag.INFO, f"batch=256, fp32 -> {dt * 1000:6.2f} ms/forward  ({256 / dt:>8,.0f} samples/s)")

    try:
        model_bf = model.to(torch.bfloat16)
        x_bf = x_fp32.to(torch.bfloat16)
        dt = _time(model_bf, x_bf)
        line(_Tag.INFO, f"batch=256, bf16 -> {dt * 1000:6.2f} ms/forward  ({256 / dt:>8,.0f} samples/s)")
    except Exception as e:
        line(_Tag.WARN, f"bf16 path failed: {e}")


# ---------- recommendation ----------
def print_recommendation(info) -> None:
    print("\n=== Recommendation ===")
    devices = info.get("devices", [])
    arch_list = set(info.get("arch_list", []))
    if not devices:
        print("  No CUDA device usable from this env. See WSL_SETUP.md step 2.")
        return

    blackwell = [d for d in devices if d["cap"] >= (12, 0)]
    if blackwell and not any(a.startswith("sm_12") for a in arch_list):
        print(textwrap.dedent("""\
              Your torch wheel was NOT built with sm_120 kernels but you have a
              Blackwell-class GPU. Training will fall back to slow PTX JIT or
              fail outright on some ops.

              Recommended fix:
                  pip install --upgrade torch torchvision \\
                      --index-url https://download.pytorch.org/whl/cu128

              (Use a fresh conda env if pip can't satisfy dependencies in the
              current one — see WSL_SETUP.md option B.)"""))
        return

    if blackwell:
        print(f"  Blackwell-ready: torch built for {' '.join(sorted(arch_list))}")
        print("  This env should fully exploit the GPU. No env switch needed.")
    else:
        non_blackwell_caps = sorted({d['sm'] for d in devices})
        print(f"  Non-Blackwell device(s) detected ({', '.join(non_blackwell_caps)}). "
              "All checks passed; nothing Blackwell-specific to do.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="run timing benchmarks (matmul + WRN-28-2)")
    args = parser.parse_args()

    check_python()
    check_nvidia_smi()
    torch, info = check_torch()
    if torch is None:
        sys.exit(1)

    if not info.get("devices"):
        print_recommendation(info)
        sys.exit(2)

    if not check_simple_forward(torch):
        print_recommendation(info)
        sys.exit(3)

    check_memory_alloc(torch)

    if args.bench:
        bench_matmul(torch)
        bench_wrn(torch)

    print_recommendation(info)


if __name__ == "__main__":
    main()
