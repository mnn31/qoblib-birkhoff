#!/usr/bin/env python3
"""Exact Birkhoff--von Neumann decomposition benchmarks for QOBLIB.

The solver preserves integer residuals throughout.  It supports two constructive
policies: maximum total residual weight (``largest_weight``) and maximum
bottleneck weight (``max_min``).  Both extract a positive perfect matching and
subtract its smallest selected residual, so the output is an exact convex
decomposition of every input matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def positive_perfect_matching(residual: np.ndarray, policy: str) -> np.ndarray:
    """Return a perfect matching using only positive entries of ``residual``."""
    n = residual.shape[0]
    largest = int(residual.max())
    forbidden = n * largest + 1

    if policy == "largest_weight":
        cost = np.where(residual > 0, -residual, forbidden)
        rows, cols = linear_sum_assignment(cost)
        assert np.all(residual[rows, cols] > 0)
        return cols

    if policy not in {
        "max_min",
        "max_min_zero_max",
        "max_min_low_sum",
        "max_min_zero_low_sum",
    }:
        raise ValueError(f"Unknown policy: {policy}")

    values = np.unique(residual[residual > 0])
    low, high = 0, len(values) - 1
    best: np.ndarray | None = None
    while low <= high:
        middle = (low + high) // 2
        threshold = values[middle]
        allowed = residual >= threshold
        cost = np.where(allowed, 0, 1)
        if policy == "max_min_zero_max":
            cost = np.where(allowed, (residual > threshold).astype(int), forbidden)
        elif policy == "max_min_low_sum":
            cost = np.where(allowed, residual, forbidden)
        elif policy == "max_min_zero_low_sum":
            zero_priority = n * largest + 1
            forbidden = (n + 1) * zero_priority + largest
            cost = np.where(
                allowed,
                (residual > threshold).astype(np.int64) * zero_priority + residual,
                forbidden,
            )
        rows, cols = linear_sum_assignment(cost)
        if np.all(allowed[rows, cols]):
            best = cols
            low = middle + 1
        else:
            high = middle - 1
    assert best is not None
    return best


def decompose(instance: dict[str, object], policy: str) -> tuple[list[int], list[list[int]]]:
    """Build an exact integer decomposition and validate its reconstruction."""
    n = int(instance["n"])
    target = np.asarray(instance["scaled_doubly_stochastic_matrix"], dtype=np.int64)
    target = target.reshape(n, n)
    residual = target.copy()
    weights: list[int] = []
    permutations: list[list[int]] = []

    while np.any(residual):
        permutation = positive_perfect_matching(residual, policy)
        chosen = residual[np.arange(n), permutation]
        weight = int(chosen.min())
        assert weight > 0
        residual[np.arange(n), permutation] -= weight
        weights.append(weight)
        permutations.append((permutation + 1).tolist())

    reconstructed = np.zeros_like(target)
    for weight, permutation in zip(weights, permutations, strict=True):
        reconstructed[np.arange(n), np.asarray(permutation) - 1] += weight
    assert np.array_equal(reconstructed, target)
    assert sum(weights) == int(instance["scale"])
    return weights, permutations


def decompose_with_trajectory(
    instance: dict[str, object], policy: str
) -> tuple[list[int], list[list[int]], list[dict[str, float | int]]]:
    """Build an exact decomposition and record the residual after every step."""
    n = int(instance["n"])
    scale = int(instance["scale"])
    target = np.asarray(instance["scaled_doubly_stochastic_matrix"], dtype=np.int64).reshape(n, n)
    residual = target.copy()
    weights: list[int] = []
    permutations: list[list[int]] = []
    trajectory: list[dict[str, float | int]] = []
    started = time.perf_counter()

    while np.any(residual):
        permutation = positive_perfect_matching(residual, policy)
        weight = int(residual[np.arange(n), permutation].min())
        assert weight > 0
        residual[np.arange(n), permutation] -= weight
        weights.append(weight)
        permutations.append((permutation + 1).tolist())
        approximation = float(np.square(residual, dtype=np.float64).sum() / (n * n * scale * scale))
        trajectory.append(
            {
                "Time": time.perf_counter() - started,
                "Number of Matrices": len(weights),
                "Approximation": approximation,
            }
        )

    assert trajectory[-1]["Approximation"] == 0.0
    return weights, permutations, trajectory


def solve_file(input_path: Path, output_path: Path, policy: str) -> None:
    payload = json.loads(input_path.read_text())
    solutions: dict[str, dict[str, object]] = {}
    for key, instance in payload.items():
        if key == "_license":
            continue
        started = time.perf_counter()
        weights, permutations = decompose(instance, policy)
        elapsed = time.perf_counter() - started
        instance_id = str(instance["id"])
        print(f"{instance_id}\tterms={len(weights)}\tseconds={elapsed:.3f}")
        solutions[key] = {
            "id": instance_id,
            "scaled_doubly_stochastic_matrix": instance["scaled_doubly_stochastic_matrix"],
            "weights": weights,
            "permutations": [entry for permutation in permutations for entry in permutation],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(solutions, separators=(",", ":")) + "\n")


SUMMARY_COLUMNS = [
    "Problem", "Submitter", "Affiliation", "Date", "Reference",
    "Best Objective Value", "Optimality Bound", "Modeling Approach",
    "# Decision Variables", "# Binary Variables", "# Integer Variables",
    "# Continuous Variables", "# Non-Zero Coefficients", "Coefficients Type",
    "Coefficients Range", "Workflow", "Algorithm Type", "Paradigm", "# Runs",
    "# Feasible Runs", "# Successful Runs", "Success Threshold",
    "Hardware Specifications", "Total Runtime", "Time to Solution", "CPU Runtime",
    "GPU Runtime", "QPU Runtime", "Other HW Runtime", "Remarks",
]


def write_submission(input_path: Path, submission_root: Path, policy: str) -> None:
    """Create one QOBLIB submission directory from an instance collection."""
    payload = json.loads(input_path.read_text())
    for instance in payload.values():
        if not isinstance(instance, dict) or "id" not in instance:
            continue
        weights, permutations, trajectory = decompose_with_trajectory(instance, policy)
        instance_id = str(instance["id"])
        n = int(instance["n"])
        elapsed = float(trajectory[-1]["Time"])
        result = {
            "id": instance_id,
            "scaled_doubly_stochastic_matrix": instance["scaled_doubly_stochastic_matrix"],
            "weights": weights,
            "permutations": [entry for permutation in permutations for entry in permutation],
        }
        qoblib_trajectory = [
            [
                {
                    "Time": point["Time"],
                    "Incumbent": point["Number of Matrices"],
                    "Error": point["Approximation"],
                }
                for point in trajectory
            ]
        ]
        instance_dir = submission_root / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        (instance_dir / f"{instance_id}_solution.json").write_text(
            json.dumps({"1": result}, indent=2) + "\n"
        )
        (instance_dir / f"{instance_id}_objective_time_series.json").write_text(
            json.dumps(qoblib_trajectory, indent=2) + "\n"
        )
        summary = {
            "Problem": instance_id,
            "Submitter": "Manan Gupta",
            "Affiliation": "Independent Researcher",
            "Date": "2026-08-17",
            "Reference": "https://github.com/mnn31/qoblib-birkhoff",
            "Best Objective Value": len(weights),
            "Optimality Bound": "N/A",
            "Modeling Approach": "Exact integer Birkhoff decomposition using bottleneck perfect matchings.",
            "# Decision Variables": len(weights) * (n + 1),
            "# Binary Variables": len(weights) * n,
            "# Integer Variables": len(weights),
            "# Continuous Variables": 0,
            "# Non-Zero Coefficients": 2 * n,
            "Coefficients Type": "Binary and integer",
            "Coefficients Range": f"1 to {int(instance['scale'])}",
            "Workflow": "At each iteration, maximize the smallest residual selected by a perfect matching. Within that threshold, maximize eliminated entries and then minimize the matching residual sum. Subtract the selected minimum exactly.",
            "Algorithm Type": "Deterministic",
            "Paradigm": "Classical",
            "# Runs": 1,
            "# Feasible Runs": 1,
            "# Successful Runs": 1,
            "Success Threshold": 0,
            "Hardware Specifications": "Apple MacBook Pro with Apple M3 Pro, 11 CPU cores, 18 GB unified memory",
            "Total Runtime": f"{elapsed:.6f}",
            "Time to Solution": f"{elapsed:.6f}",
            "CPU Runtime": f"{elapsed:.6f}",
            "GPU Runtime": 0,
            "QPU Runtime": 0,
            "Other HW Runtime": 0,
            "Remarks": "Exact reconstruction verified; no optimality claim is made.",
        }
        with (instance_dir / f"{instance_id}_summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument(
        "--submission-root",
        type=Path,
        help="Write QOBLIB submission files below this directory.",
    )
    parser.add_argument(
        "--policy",
        choices=(
            "largest_weight",
            "max_min",
            "max_min_zero_max",
            "max_min_low_sum",
            "max_min_zero_low_sum",
        ),
        default="largest_weight",
    )
    arguments = parser.parse_args()
    if arguments.submission_root is not None:
        write_submission(arguments.input, arguments.submission_root, arguments.policy)
    elif arguments.output is not None:
        solve_file(arguments.input, arguments.output, arguments.policy)
    else:
        parser.error("output is required unless --submission-root is used")


if __name__ == "__main__":
    main()
