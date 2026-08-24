#!/usr/bin/env python3
"""Benchmark routed backward GGTensile and HIP kernels."""

from benchmark_kernel_runner import run_kernels

if __name__ == "__main__":
    run_kernels("GroupedBackward", routed=True)
