#!/usr/bin/env python3
"""Benchmark the paired routed backward public API against AITER GMM."""

from benchmark_api_runner import run_api

if __name__ == "__main__":
    run_api("GroupedBackwardPair", routed=True)
