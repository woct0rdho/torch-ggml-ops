#!/usr/bin/env python3
"""Benchmark the ordinary backward public API against torch.mm."""

from benchmark_api_runner import run_api

if __name__ == "__main__":
    run_api("OrdinaryBackward", routed=False)
