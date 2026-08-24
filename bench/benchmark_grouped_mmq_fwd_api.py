#!/usr/bin/env python3
"""Benchmark the routed forward public API against AITER GMM."""

from benchmark_api_runner import run_api

if __name__ == "__main__":
    run_api("GroupedForward", routed=True)
