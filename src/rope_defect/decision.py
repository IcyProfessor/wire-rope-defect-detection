from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ACTIONS = {
    "CRITICAL_DEFECT": "Немедленный визуальный осмотр и повторная диагностика участка.",
    "CONFIRMED_DEFECT": "Записать подтвержденный дефект, назначить внеочередной контроль.",
    "VISUAL_SUSPECT": "Проверить участок оператором.",
    "MAGNETIC_SUSPECT": "Вероятный внутренний дефект, повторить магнитный контроль.",
    "NORMAL": "Продолжить штатный мониторинг.",
}

GT_TYPES = {"broken_wire", "corrosion", "kink_or_bend", "normal"}


def unit_score(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return value


@dataclass(frozen=True)
class DecisionConfig:
    visual_threshold: float = 0.35
    magnetic_threshold: float = 0.55
    camera_weight: float = 0.60
    hall_weight: float = 0.40
    critical_threshold: float = 0.75

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            unit_score(value, name)
        if not math.isclose(self.camera_weight + self.hall_weight, 1.0, abs_tol=1e-12):
            raise ValueError("Decision weights must sum to one")


def load_decision_config(path: Path | None = None) -> DecisionConfig:
    if path is None:
        return DecisionConfig()

    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Decision config must be a mapping")

    expected = set(asdict(DecisionConfig()))
    if set(obj) != expected:
        raise ValueError(f"Decision config must contain exactly: {sorted(expected)}")

    return DecisionConfig(**{key: float(value) for key, value in obj.items()})


def decide(
    camera_conf: float,
    hall_score: float,
    cfg: DecisionConfig = DecisionConfig(),
) -> dict[str, Any]:
    camera_conf = unit_score(camera_conf, "camera_conf")
    hall_score = unit_score(hall_score, "hall_score")

    visual = camera_conf >= cfg.visual_threshold
    magnetic = hall_score >= cfg.magnetic_threshold
    risk = cfg.camera_weight * camera_conf + cfg.hall_weight * hall_score

    if visual and magnetic and risk >= cfg.critical_threshold:
        status, level = "CRITICAL_DEFECT", "HIGH"
    elif visual and magnetic:
        status, level = "CONFIRMED_DEFECT", "HIGH"
    elif visual:
        status, level = "VISUAL_SUSPECT", "MEDIUM"
    elif magnetic:
        status, level = "MAGNETIC_SUSPECT", "MEDIUM"
    else:
        status, level = "NORMAL", "LOW"

    return {
        "visual_defect": int(visual),
        "magnetic_anomaly": int(magnetic),
        "risk_score": round(risk, 4),
        "decision": status,
        "risk_level": level,
        "recommended_action": ACTIONS[status],
    }


def gt_type_from_name(name: str) -> str:
    for prefix in ("broken_wire", "corrosion", "kink_or_bend", "normal"):
        if name.startswith(prefix + "_"):
            return prefix
    raise ValueError(f"Unknown class prefix: {name}")


def simulate_hall_score(
    gt_type: str,
    camera_conf: float,
    rng: np.random.Generator,
) -> float:
    """Synthetic magnetic anomaly score used only by the demonstration pipeline."""
    distributions = {
        "broken_wire": (0.82, 0.10),
        "corrosion": (0.45, 0.14),
        "kink_or_bend": (0.62, 0.12),
        "normal": (0.18, 0.10),
    }

    if gt_type not in distributions:
        raise ValueError(f"Unknown simulated class: {gt_type}")

    camera_conf = unit_score(camera_conf, "camera_conf")
    mean, std = distributions[gt_type]
    return float(
        np.clip(
            0.75 * rng.normal(mean, std) + 0.25 * camera_conf,
            0.0,
            1.0,
        )
    )


def simulate_journal(
    camera_rows: list[dict[str, Any]],
    seed: int = 42,
    cfg: DecisionConfig = DecisionConfig(),
) -> list[dict[str, Any]]:
    if not camera_rows:
        raise ValueError("No camera observations")

    names = [str(row["image_name"]) for row in camera_rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image names in camera observations")

    rng = np.random.default_rng(seed)
    result = []

    for index, row in enumerate(sorted(camera_rows, key=lambda item: item["image_name"])):
        gt_type = str(row["gt_type"])
        gt_defect = float(row["gt_defect"])

        if gt_type not in GT_TYPES or gt_defect not in (0, 1):
            raise ValueError("Invalid ground-truth values")
        if int(gt_defect) != int(gt_type != "normal"):
            raise ValueError("Ground-truth class and binary label disagree")

        raw_conf = unit_score(row["camera_conf"], "camera_conf")
        camera_conf = round(raw_conf, 4)
        camera_defect = int(raw_conf >= cfg.visual_threshold)

        position = round(10.0 + index * 0.75 + rng.normal(0, 0.03), 2)
        hall_score = simulate_hall_score(gt_type, camera_conf, rng)
        decision = decide(camera_conf, hall_score, cfg)

        result.append({
            "event_id": index + 1,
            "image_name": str(row["image_name"]),
            "rope_position_m": position,
            "gt_type": gt_type,
            "gt_defect": int(gt_defect),
            "camera_conf": camera_conf,
            "camera_defect": camera_defect,
            "hall_score": round(hall_score, 4),
            "magnetic_anomaly": decision["magnetic_anomaly"],
            "risk_score": decision["risk_score"],
            "decision": decision["decision"],
            "risk_level": decision["risk_level"],
            "recommended_action": decision["recommended_action"],
        })

    return result
