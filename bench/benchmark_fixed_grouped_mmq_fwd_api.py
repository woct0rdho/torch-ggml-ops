#!/usr/bin/env python3
"""Benchmark the fixed-group forward public API against torch.bmm."""

from benchmark_api_runner import run_api

if __name__ == "__main__":
    run_api("FixedGroupedForward", routed=False)
