"""Image-level metrics and diagnostic journal summaries."""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from .decision import ACTIONS, unit_score


def binary_metrics(truth: Sequence[int], predicted: Sequence[int]) -> dict[str, Any]:
    if not truth or len(truth) != len(predicted):
        raise ValueError("Truth and prediction arrays must have equal, nonzero length")
    if any(value not in (0, 1) for value in [*truth, *predicted]):
        raise ValueError("Only binary labels 0 and 1 are allowed")

    tp = sum(t == 1 and p == 1 for t, p in zip(truth, predicted))
    tn = sum(t == 0 and p == 0 for t, p in zip(truth, predicted))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, predicted))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, predicted))

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "n_images": len(truth),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "accuracy": (tp + tn) / len(truth),
        "specificity": ratio(tn, tn + fp),
        "false_positive_rate": ratio(fp, fp + tn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def journal_summary(
    rows: list[dict[str, Any]],
    threshold: float = 0.35,
) -> dict[str, Any]:
    threshold = unit_score(threshold, "threshold")
    truth: list[int] = []
    visual: list[int] = []
    system: list[int] = []

    for row in rows:
        gt_defect = float(row["gt_defect"])
        if gt_defect not in (0, 1):
            raise ValueError("Invalid gt_defect")
        if row["decision"] not in ACTIONS:
            raise ValueError(f"Unknown decision state: {row['decision']}")

        truth.append(int(gt_defect))
        visual.append(int(unit_score(row["camera_conf"], "camera_conf") >= threshold))
        system.append(int(row["decision"] != "NORMAL"))

    return {
        "metric_unit": "image",
        "visual_threshold": threshold,
        "vision_only": binary_metrics(truth, visual),
        "system_alert": binary_metrics(truth, system),
        "decision_counts": dict(Counter(row["decision"] for row in rows)),
    }
