"""Ultralytics adapter for training, inference and evaluation."""
from __future__ import annotations
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .common import new_output_directory, sha256_file, write_json
from .dataset import Dataset, IMAGE_EXTENSIONS, load_dataset, parse_label
from .decision import gt_type_from_name, unit_score
from .reporting import binary_metrics


def backend():
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is unavailable. Install the optional backend with: pip install -e '.[yolo]'") from exc
    return YOLO


def environment() -> dict[str, str | None]:
    import platform
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("ultralytics", "torch", "torchvision", "numpy", "Pillow", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_yaml(dataset: Dataset, output: Path) -> Path:
    """Resolve dataset paths relative to its YAML, not the caller's working directory."""
    cfg = {"path": str(dataset.root), **{k: str(v) for k, v in dataset.image_dirs.items()}, "names": dataset.names}
    dest = output / "resolved_data.yaml"
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return dest


def training_plan(data: Path, config: Path, output: Path, device: str = "cpu",
                  initial_weights: str | None = None) -> dict[str, Any]:
    dataset = load_dataset(data)
    spec = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("train"), dict):
        raise ValueError("Training configuration requires a train mapping")
    if int(spec["num_classes"]) != len(dataset.names):
        raise ValueError("Configuration and dataset class counts disagree")
    for split in ("train", "val", "test"):
        for image in dataset.images(split):
            parse_label(dataset.label_path(image), dataset.names)
    kwargs = dict(spec["train"])
    forbidden = {"data", "project", "name", "device", "model"} & kwargs.keys()
    if forbidden:
        raise ValueError(f"Runtime keys belong outside train config: {sorted(forbidden)}")
    out = output.resolve()
    kwargs.update(data=str(dataset.yaml_path), project=str(out), name="training", device=device)
    return {"initial_model": initial_weights or spec["initial_model"],
            "num_classes": len(dataset.names), "kwargs": kwargs,
            "note": "At execution data is replaced by an output-local YAML with absolute paths."}


def train(data: Path, config: Path, output: Path, device: str, initial_weights: str | None = None) -> Path:
    plan = training_plan(data, config, output, device, initial_weights)
    YOLO = backend()
    output = new_output_directory(output)
    plan["kwargs"]["data"] = str(runtime_yaml(load_dataset(data), output))
    write_json(output / "training_request.json", {**plan, "environment": environment()})
    model = YOLO(plan["initial_model"])
    model.train(**plan["kwargs"])
    return output


def load_model(weights: Path):
    if not weights.is_file():
        raise FileNotFoundError(f"Model file is missing: {weights}. Provide a trusted local checkpoint.")
    return backend()(str(weights.resolve()))


def collect_predictions(model: Any, images: list[Path], output: Path, device: str,
                        confidence: float = 0.05, iou: float = 0.5) -> list[dict[str, Any]]:
    unit_score(confidence, "confidence")
    unit_score(iou, "iou")
    if not images:
        raise ValueError("No images to predict")
    results = model.predict(source=[str(p) for p in sorted(images)], imgsz=640, conf=confidence,
                            iou=iou, save=True, save_txt=True, save_conf=True,
                            project=str(output), name="visualizations", device=device,
                            verbose=False, stream=True)
    rows = []
    for res in results:
        boxes = []
        if res.boxes is not None:
            for xyxy, conf, cls in zip(res.boxes.xyxy.cpu().tolist(), res.boxes.conf.cpu().tolist(), res.boxes.cls.cpu().tolist()):
                boxes.append({"class_id": int(cls), "confidence": float(conf), "xyxy": [float(v) for v in xyxy]})
        max_conf = max((b["confidence"] for b in boxes), default=0.0)
        rows.append({"image_name": Path(res.path).name,
                     "camera_conf": round(max_conf, 4), "camera_conf_raw": max_conf,
                     "num_camera_detections": len(boxes), "boxes": boxes})
    return sorted(rows, key=lambda r: r["image_name"])


def predict(weights: Path, source: Path, output: Path, device: str, confidence: float, iou: float) -> Path:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        images = [source]
    elif source.is_dir():
        images = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        raise ValueError("source must be an image or an image directory")
    if len({p.name for p in images}) != len(images):
        raise ValueError("Input image basenames must be unique")
    if not images:
        raise ValueError("No supported images")
    model = load_model(weights)
    output = new_output_directory(output)
    rows = collect_predictions(model, images, output, device, confidence, iou)
    write_json(output / "predictions.json", {"source": "live_yolo_inference", "environment": environment(),
                                             "weights_sha256": sha256_file(weights),
                                             "imgsz": 640, "confidence_floor": confidence, "nms_iou": iou,
                                             "model_names": model.names, "images": rows})
    return output


def evaluate(weights: Path, data: Path, split: str, output: Path, device: str, threshold: float) -> Path:
    threshold = unit_score(threshold, "threshold")
    if threshold < 0.05:
        raise ValueError("Image decision threshold must be >= prediction retention floor 0.05")
    dataset = load_dataset(data)
    images = dataset.images(split)
    truth = {p.name: int(bool(parse_label(dataset.label_path(p), dataset.names))) for p in images}
    model = load_model(weights)
    if dict(model.names) != dataset.names:
        raise ValueError("Model and dataset class names differ. Prepare binary labels for the binary model.")
    output = new_output_directory(output)
    resolved = runtime_yaml(dataset, output)
    metrics = model.val(data=str(resolved), split=split, imgsz=640, device=device,
                        conf=0.001, iou=0.7, plots=True, project=str(output), name="object_detection")
    rows = collect_predictions(model, images, output, device, confidence=0.05, iou=0.5)
    targets = [truth[r["image_name"]] for r in rows]
    flags = [int(r["camera_conf_raw"] >= threshold) for r in rows]
    write_json(output / "predictions.json", {"source": "live_yolo_inference", "environment": environment(),
                                             "weights_sha256": sha256_file(weights), "split": split,
                                             "imgsz": 640, "confidence_floor": 0.05, "nms_iou": 0.5,
                                             "model_names": model.names, "images": rows})
    result = {"split": split, "environment": environment(), "weights_sha256": sha256_file(weights),
              "object_detection": {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                                   "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
                                   "prediction_floor": 0.001, "nms_iou": 0.7,
                                   "threshold_note": "Ultralytics summary P/R is not the fixed image-level operating point."},
              "image_level": {"threshold": threshold, "prediction_floor": 0.05, "nms_iou": 0.5,
                              **binary_metrics(targets, flags)},
              "warning": "Do not tune decision thresholds on the test split."}
    write_json(output / "metrics.json", result)
    return output


def camera_rows_with_ground_truth(predictions: Path, data: Path, split: str,
                                  threshold: float = 0.35) -> list[dict[str, Any]]:
    """Join labels for simulation/evaluation only; predict() remains label-free."""
    obj = json.loads(predictions.read_text(encoding="utf-8"))
    if obj.get("confidence_floor", 1.0) > threshold:
        raise ValueError("Stored predictions discard scores below the decision threshold")
    dataset = load_dataset(data)
    labels = {p.name: dataset.label_path(p) for p in dataset.images(split)}
    observations = obj["images"]
    names = [row["image_name"] for row in observations]
    if len(names) != len(set(names)) or set(names) != set(labels):
        raise ValueError("Prediction filenames must match the selected split exactly")
    rows = []
    for row in observations:
        name = row["image_name"]
        kind = gt_type_from_name(name)
        rows.append({"image_name": name, "gt_type": kind,
                     "gt_defect": int(bool(parse_label(labels[name], dataset.names))),
                     "camera_conf": row["camera_conf"],
                     "camera_defect": int(row.get("camera_conf_raw", row["camera_conf"]) >= threshold)})
    return rows
