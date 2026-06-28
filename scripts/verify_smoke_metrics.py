#!/usr/bin/env python3
"""Run the deterministic NuFuse smoke test and compare terminal metrics to a hardcoded reference."""

import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# Reference metrics captured from a canonical deterministic CPU smoke test run:
# TORCH_COMPILE_DISABLE=1 uv run -m lsp_detr +experiment=NuFuseSmoke +trainer.accelerator=cpu
# Config: seed=42, no augmentations, enable_progress_bar=false, bf16-mixed on CPU.
REFERENCE_METRICS: dict[str, dict[str, dict[str, float]]] = {
    "train": {
        "0": {
            "train/loss_ce": 0.547151,
            "train/loss_ce_0": 0.578954,
            "train/loss_ce_1": 0.567479,
            "train/loss_ce_2": 0.573062,
            "train/loss_ce_3": 0.568918,
            "train/loss_ce_4": 0.570245,
            "train/loss_centroids": 0.037244,
            "train/loss_centroids_0": 0.037272,
            "train/loss_centroids_1": 0.037244,
            "train/loss_centroids_2": 0.037272,
            "train/loss_centroids_3": 0.037244,
            "train/loss_centroids_4": 0.037276,
            "train/loss_radial_distances": 0.569717,
            "train/loss_radial_distances_0": 0.569617,
            "train/loss_radial_distances_1": 0.569717,
            "train/loss_radial_distances_2": 0.569617,
            "train/loss_radial_distances_3": 0.569717,
            "train/loss_radial_distances_4": 0.569911,
        },
        "1": {
            "train/loss_ce": 0.595259,
            "train/loss_ce_0": 0.654145,
            "train/loss_ce_1": 0.638601,
            "train/loss_ce_2": 0.650043,
            "train/loss_ce_3": 0.621942,
            "train/loss_ce_4": 0.629013,
            "train/loss_centroids": 0.017243,
            "train/loss_centroids_0": 0.017156,
            "train/loss_centroids_1": 0.017194,
            "train/loss_centroids_2": 0.017177,
            "train/loss_centroids_3": 0.017508,
            "train/loss_centroids_4": 0.017523,
            "train/loss_radial_distances": 0.606419,
            "train/loss_radial_distances_0": 0.606419,
            "train/loss_radial_distances_1": 0.606419,
            "train/loss_radial_distances_2": 0.606419,
            "train/loss_radial_distances_3": 0.606419,
            "train/loss_radial_distances_4": 0.606419,
        },
        "2": {
            "train/loss_ce": 0.58349,
            "train/loss_ce_0": 0.618154,
            "train/loss_ce_1": 0.614731,
            "train/loss_ce_2": 0.641132,
            "train/loss_ce_3": 0.608841,
            "train/loss_ce_4": 0.611691,
            "train/loss_centroids": 0.018199,
            "train/loss_centroids_0": 0.018181,
            "train/loss_centroids_1": 0.018198,
            "train/loss_centroids_2": 0.018199,
            "train/loss_centroids_3": 0.018418,
            "train/loss_centroids_4": 0.018502,
            "train/loss_radial_distances": 0.608578,
            "train/loss_radial_distances_0": 0.608016,
            "train/loss_radial_distances_1": 0.608089,
            "train/loss_radial_distances_2": 0.608015,
            "train/loss_radial_distances_3": 0.608723,
            "train/loss_radial_distances_4": 0.608233,
        },
        "3": {
            "train/loss_ce": 0.584737,
            "train/loss_ce_0": 0.63621,
            "train/loss_ce_1": 0.615638,
            "train/loss_ce_2": 0.63215,
            "train/loss_ce_3": 0.605041,
            "train/loss_ce_4": 0.60078,
            "train/loss_centroids": 0.033704,
            "train/loss_centroids_0": 0.034382,
            "train/loss_centroids_1": 0.034382,
            "train/loss_centroids_2": 0.034382,
            "train/loss_centroids_3": 0.033673,
            "train/loss_centroids_4": 0.033874,
            "train/loss_radial_distances": 0.585418,
            "train/loss_radial_distances_0": 0.582223,
            "train/loss_radial_distances_1": 0.582195,
            "train/loss_radial_distances_2": 0.582164,
            "train/loss_radial_distances_3": 0.585927,
            "train/loss_radial_distances_4": 0.58482,
        },
        "4": {
            "train/loss_ce": 0.540783,
            "train/loss_ce_0": 0.599267,
            "train/loss_ce_1": 0.576098,
            "train/loss_ce_2": 0.589399,
            "train/loss_ce_3": 0.56287,
            "train/loss_ce_4": 0.558949,
            "train/loss_centroids": 0.017667,
            "train/loss_centroids_0": 0.017186,
            "train/loss_centroids_1": 0.017177,
            "train/loss_centroids_2": 0.017199,
            "train/loss_centroids_3": 0.017696,
            "train/loss_centroids_4": 0.017668,
            "train/loss_radial_distances": 0.606338,
            "train/loss_radial_distances_0": 0.606402,
            "train/loss_radial_distances_1": 0.606389,
            "train/loss_radial_distances_2": 0.606376,
            "train/loss_radial_distances_3": 0.606361,
            "train/loss_radial_distances_4": 0.606346,
        },
    },
    "validation": {
        "0": {
            "validation/loss_ce": 0.583399,
            "validation/loss_ce_0": 0.623531,
            "validation/loss_ce_1": 0.607748,
            "validation/loss_ce_2": 0.620771,
            "validation/loss_ce_3": 0.600524,
            "validation/loss_ce_4": 0.607219,
            "validation/loss_centroids": 0.027692,
            "validation/loss_centroids_0": 0.028602,
            "validation/loss_centroids_1": 0.028608,
            "validation/loss_centroids_2": 0.028581,
            "validation/loss_centroids_3": 0.027785,
            "validation/loss_centroids_4": 0.027786,
            "validation/loss_radial_distances": 0.589929,
            "validation/loss_radial_distances_0": 0.586506,
            "validation/loss_radial_distances_1": 0.586506,
            "validation/loss_radial_distances_2": 0.586602,
            "validation/loss_radial_distances_3": 0.589929,
            "validation/loss_radial_distances_4": 0.589929,
            "validation/pq": 0.013088,
        },
        "4": {
            "validation/loss_ce": 0.514471,
            "validation/loss_ce_0": 0.590548,
            "validation/loss_ce_1": 0.566546,
            "validation/loss_ce_2": 0.571302,
            "validation/loss_ce_3": 0.543997,
            "validation/loss_ce_4": 0.543672,
            "validation/loss_centroids": 0.02764,
            "validation/loss_centroids_0": 0.028601,
            "validation/loss_centroids_1": 0.028607,
            "validation/loss_centroids_2": 0.028587,
            "validation/loss_centroids_3": 0.028712,
            "validation/loss_centroids_4": 0.027755,
            "validation/loss_radial_distances": 0.588984,
            "validation/loss_radial_distances_0": 0.586368,
            "validation/loss_radial_distances_1": 0.586247,
            "validation/loss_radial_distances_2": 0.586213,
            "validation/loss_radial_distances_3": 0.586097,
            "validation/loss_radial_distances_4": 0.589097,
            "validation/pq": 0.011258,
        },
    },
}


def parse_output(text: str) -> dict[tuple[int, str], dict[str, float]]:
    """Parse clean 'Epoch N - phase: name=value ...' lines from captured terminal output."""
    data: dict[tuple[int, str], dict[str, float]] = {}
    for raw in text.splitlines():
        line = raw.split("\r")[-1]
        match = re.match(r"Epoch (\d+) - (train|validation):\s*(.*)", line)
        if not match:
            continue
        epoch = int(match.group(1))
        phase = match.group(2)
        tail = match.group(3)
        metrics = {
            name: float(value) for name, value in re.findall(r"([\w_]+)=([\d.]+)", tail)
        }
        data[(epoch, phase)] = metrics
    return data


def run_smoke_test() -> str:
    """Run the deterministic smoke test, stream output, and return captured output."""
    env = os.environ.copy()
    env["TORCH_COMPILE_DISABLE"] = "1"

    cmd = [
        "uv",
        "run",
        "-m",
        "lsp_detr",
        "+experiment=NuFuseSmoke",
        "+trainer.accelerator=cpu",
    ]

    print(f"Running smoke test from {REPO_ROOT}")
    print(f"Command: {' '.join(cmd)}")
    print("---")

    process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    lines: list[str] = []
    if process.stdout is not None:
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            lines.append(line)

    process.wait()
    output = "".join(lines)

    print("---")
    if process.returncode != 0:
        print(f"Smoke test failed with exit code: {process.returncode}")
        raise SystemExit(1)

    print(f"Smoke test finished successfully (exit code {process.returncode}).")
    return output


def main() -> int:
    output = run_smoke_test()
    rerun = parse_output(output)

    print(f"Parsed {len(rerun)} epoch summary line(s) from terminal output.")

    mismatches: list[tuple[str, int, str, float, float | None, object]] = []
    checked = 0
    for phase in ("train", "validation"):
        ref_phase = REFERENCE_METRICS.get(phase, {})
        for epoch_str, metrics in ref_phase.items():
            epoch = int(epoch_str)
            run_metrics = rerun.get((epoch, phase), {})
            print(
                f"Checking {phase} epoch {epoch}: "
                f"{len(run_metrics)} metric(s) parsed, {len(metrics)} in reference"
            )
            for full_name, expected in metrics.items():
                checked += 1
                _, short_name = full_name.split("/", 1)
                actual = run_metrics.get(short_name)
                if actual is None:
                    mismatches.append(
                        (phase, epoch, short_name, expected, None, "missing")
                    )
                elif actual != expected:
                    mismatches.append(
                        (
                            phase,
                            epoch,
                            short_name,
                            expected,
                            actual,
                            abs(actual - expected),
                        )
                    )

    print(f"Total metrics checked: {checked}")

    if mismatches:
        print("MISMATCHES between run and reference:")
        for phase, epoch, name, expected, actual, diff in mismatches:
            print(
                f"  {phase} epoch {epoch} {name}: "
                f"expected {expected}, got {actual} (diff {diff})"
            )
        return 1

    print("All terminal metrics match the reference exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
