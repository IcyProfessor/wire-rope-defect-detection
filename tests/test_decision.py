import math
from pathlib import Path

import pytest

from rope_defect.common import read_csv
from rope_defect.decision import (
    DecisionConfig,
    decide,
    load_decision_config,
    simulate_journal,
)
from rope_defect.reporting import binary_metrics, journal_summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "camera,hall,state",
    [
        (0.0, 0.0, "NORMAL"),
        (0.35, 0.54, "VISUAL_SUSPECT"),
        (0.34, 0.55, "MAGNETIC_SUSPECT"),
        (0.35, 0.55, "CONFIRMED_DEFECT"),
        (0.9, 0.9, "CRITICAL_DEFECT"),
    ],
)
def test_states(camera, hall, state):
    assert decide(camera, hall)["decision"] == state


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf])
def test_invalid_scores_rejected(value):
    with pytest.raises(ValueError):
        decide(value, 0.5)
    with pytest.raises(ValueError):
        decide(0.5, value)


def test_weights_sum_to_one():
    with pytest.raises(ValueError):
        DecisionConfig(camera_weight=0.7, hall_weight=0.4)


def test_risk_formula():
    assert decide(0.5, 0.7)["risk_score"] == 0.58


def test_recorded_metrics_are_image_level():
    rows = read_csv(ROOT / "artifacts/decision/diagnostic_journal.csv")
    report = journal_summary(rows)

    assert {
        key: report["vision_only"][key]
        for key in ["tp", "tn", "fp", "fn"]
    } == {"tp": 11, "tn": 3, "fp": 1, "fn": 1}
    assert report["vision_only"]["recall"] == pytest.approx(11 / 12)
    assert report["vision_only"]["accuracy"] == 14 / 16
    assert report["decision_counts"] == {
        "CONFIRMED_DEFECT": 9,
        "CRITICAL_DEFECT": 1,
        "VISUAL_SUSPECT": 2,
        "NORMAL": 4,
    }


def test_simulation_is_seeded():
    camera_rows = [
        {
            "image_name": "broken_wire_001.jpg",
            "gt_type": "broken_wire",
            "gt_defect": 1,
            "camera_conf": 0.8,
        },
        {
            "image_name": "normal_001.jpg",
            "gt_type": "normal",
            "gt_defect": 0,
            "camera_conf": 0.1,
        },
    ]

    assert simulate_journal(camera_rows, 42) == simulate_journal(camera_rows, 42)
    assert simulate_journal(camera_rows, 42) != simulate_journal(camera_rows, 43)


def test_undefined_metric_is_none():
    assert binary_metrics([0, 0], [0, 0])["precision"] is None
    assert binary_metrics([0, 0], [0, 0])["recall"] is None


def test_bad_metric_inputs():
    for truth, predicted in [([], []), ([1], []), ([2], [1])]:
        with pytest.raises(ValueError):
            binary_metrics(truth, predicted)


def test_config_file_matches_defaults():
    assert load_decision_config(
        ROOT / "configs/decision.yaml"
    ) == DecisionConfig()
