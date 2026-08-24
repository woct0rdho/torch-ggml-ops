#!/usr/bin/env python3
"""Benchmark routed forward GGTensile and HIP multiply kernels."""

from benchmark_kernel_runner import run_kernels

if __name__ == "__main__":
    run_kernels("GroupedForward", routed=True)
