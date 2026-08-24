#!/usr/bin/env python3
"""Benchmark the ordinary forward public API against torch.mm."""

from benchmark_api_runner import run_api

if __name__ == "__main__":
    run_api("OrdinaryForward", routed=False)
