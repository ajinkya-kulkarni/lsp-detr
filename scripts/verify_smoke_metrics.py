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
            "train/loss_ce": 0.500839,
            "train/loss_ce_0": 0.46859,
            "train/loss_ce_1": 0.440408,
            "train/loss_ce_2": 0.47192,
            "train/loss_ce_3": 0.491852,
            "train/loss_ce_4": 0.516975,
            "train/loss_centroids": 0.017325,
            "train/loss_centroids_0": 0.017156,
            "train/loss_centroids_1": 0.017156,
            "train/loss_centroids_2": 0.017213,
            "train/loss_centroids_3": 0.017287,
            "train/loss_centroids_4": 0.017275,
            "train/loss_radial_distances": 0.573703,
            "train/loss_radial_distances_0": 0.572934,
            "train/loss_radial_distances_1": 0.572934,
            "train/loss_radial_distances_2": 0.572934,
            "train/loss_radial_distances_3": 0.573703,
            "train/loss_radial_distances_4": 0.573703,
        },
        "1": {
            "train/loss_ce": 0.490697,
            "train/loss_ce_0": 0.463831,
            "train/loss_ce_1": 0.423595,
            "train/loss_ce_2": 0.450276,
            "train/loss_ce_3": 0.467589,
            "train/loss_ce_4": 0.501635,
            "train/loss_centroids": 0.041026,
            "train/loss_centroids_0": 0.041026,
            "train/loss_centroids_1": 0.041026,
            "train/loss_centroids_2": 0.041026,
            "train/loss_centroids_3": 0.041026,
            "train/loss_centroids_4": 0.041026,
            "train/loss_radial_distances": 0.549437,
            "train/loss_radial_distances_0": 0.549438,
            "train/loss_radial_distances_1": 0.549438,
            "train/loss_radial_distances_2": 0.549438,
            "train/loss_radial_distances_3": 0.549437,
            "train/loss_radial_distances_4": 0.549437,
        },
        "2": {
            "train/loss_ce": 0.537847,
            "train/loss_ce_0": 0.551979,
            "train/loss_ce_1": 0.518833,
            "train/loss_ce_2": 0.56024,
            "train/loss_ce_3": 0.561818,
            "train/loss_ce_4": 0.570488,
            "train/loss_centroids": 0.022253,
            "train/loss_centroids_0": 0.022821,
            "train/loss_centroids_1": 0.022821,
            "train/loss_centroids_2": 0.021885,
            "train/loss_centroids_3": 0.022137,
            "train/loss_centroids_4": 0.022128,
            "train/loss_radial_distances": 0.564761,
            "train/loss_radial_distances_0": 0.566199,
            "train/loss_radial_distances_1": 0.566199,
            "train/loss_radial_distances_2": 0.564762,
            "train/loss_radial_distances_3": 0.564762,
            "train/loss_radial_distances_4": 0.564762,
        },
        "3": {
            "train/loss_ce": 0.516136,
            "train/loss_ce_0": 0.526266,
            "train/loss_ce_1": 0.491446,
            "train/loss_ce_2": 0.518676,
            "train/loss_ce_3": 0.549173,
            "train/loss_ce_4": 0.561335,
            "train/loss_centroids": 0.016971,
            "train/loss_centroids_0": 0.017156,
            "train/loss_centroids_1": 0.017156,
            "train/loss_centroids_2": 0.017192,
            "train/loss_centroids_3": 0.016938,
            "train/loss_centroids_4": 0.016938,
            "train/loss_radial_distances": 0.573698,
            "train/loss_radial_distances_0": 0.572931,
            "train/loss_radial_distances_1": 0.57293,
            "train/loss_radial_distances_2": 0.57293,
            "train/loss_radial_distances_3": 0.5737,
            "train/loss_radial_distances_4": 0.5737,
        },
        "4": {
            "train/loss_ce": 0.471043,
            "train/loss_ce_0": 0.463603,
            "train/loss_ce_1": 0.431915,
            "train/loss_ce_2": 0.45741,
            "train/loss_ce_3": 0.478264,
            "train/loss_ce_4": 0.496778,
            "train/loss_centroids": 0.016932,
            "train/loss_centroids_0": 0.017156,
            "train/loss_centroids_1": 0.017156,
            "train/loss_centroids_2": 0.017183,
            "train/loss_centroids_3": 0.016937,
            "train/loss_centroids_4": 0.017076,
            "train/loss_radial_distances": 0.573667,
            "train/loss_radial_distances_0": 0.572918,
            "train/loss_radial_distances_1": 0.57291,
            "train/loss_radial_distances_2": 0.572906,
            "train/loss_radial_distances_3": 0.573675,
            "train/loss_radial_distances_4": 0.573674,
        },
    },
    "validation": {
        "0": {
            "validation/bPQ": 0.012056,
            "validation/loss_ce": 0.544167,
            "validation/loss_ce_0": 0.540687,
            "validation/loss_ce_1": 0.516055,
            "validation/loss_ce_2": 0.546198,
            "validation/loss_ce_3": 0.559632,
            "validation/loss_ce_4": 0.574318,
            "validation/loss_centroids": 0.027801,
            "validation/loss_centroids_0": 0.028503,
            "validation/loss_centroids_1": 0.028503,
            "validation/loss_centroids_2": 0.028455,
            "validation/loss_centroids_3": 0.027734,
            "validation/loss_centroids_4": 0.027753,
            "validation/loss_radial_distances": 0.575127,
            "validation/loss_radial_distances_0": 0.575051,
            "validation/loss_radial_distances_1": 0.575051,
            "validation/loss_radial_distances_2": 0.575468,
            "validation/loss_radial_distances_3": 0.575448,
            "validation/loss_radial_distances_4": 0.575448,
        },
        "4": {
            "validation/bPQ": 0.006181,
            "validation/loss_ce": 0.475864,
            "validation/loss_ce_0": 0.505432,
            "validation/loss_ce_1": 0.471493,
            "validation/loss_ce_2": 0.495972,
            "validation/loss_ce_3": 0.503121,
            "validation/loss_ce_4": 0.510785,
            "validation/loss_centroids": 0.027734,
            "validation/loss_centroids_0": 0.028513,
            "validation/loss_centroids_1": 0.02843,
            "validation/loss_centroids_2": 0.028109,
            "validation/loss_centroids_3": 0.027724,
            "validation/loss_centroids_4": 0.027665,
            "validation/loss_radial_distances": 0.574364,
            "validation/loss_radial_distances_0": 0.574915,
            "validation/loss_radial_distances_1": 0.5752,
            "validation/loss_radial_distances_2": 0.575572,
            "validation/loss_radial_distances_3": 0.575041,
            "validation/loss_radial_distances_4": 0.574806,
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
                    mismatches.append((phase, epoch, short_name, expected, None, "missing"))
                elif actual != expected:
                    mismatches.append(
                        (phase, epoch, short_name, expected, actual, abs(actual - expected))
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
