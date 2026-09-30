"""Utilities for YOLO datasets used by the project."""
from __future__ import annotations

import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from .common import sha256_file

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Dataset:
    yaml_path: Path
    root: Path
    names: dict[int, str]
    image_dirs: dict[str, Path]

    def images(self, split: str) -> list[Path]:
        folder = self.image_dirs[split]
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing {split} image directory: {folder}")
        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise ValueError(f"No images in {folder}")
        return files

    def label_path(self, image: Path) -> Path:
        relative = image.relative_to(self.root)
        parts = list(relative.parts)
        positions = [i for i, part in enumerate(parts) if part == "images"]
        if len(positions) != 1:
            raise ValueError(f"Expected one images component in path: {relative}")
        parts[positions[0]] = "labels"
        return self.root.joinpath(*parts).with_suffix(".txt")


def load_dataset(path: Path) -> Dataset:
    path = path.resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError("Dataset YAML must be a mapping")

    root = Path(cfg.get("path", ".")).expanduser()
    root = (root if root.is_absolute() else path.parent / root).resolve()

    raw_names = cfg.get("names")
    if isinstance(raw_names, list):
        names = dict(enumerate(map(str, raw_names)))
    elif isinstance(raw_names, dict):
        names = {int(k): str(v) for k, v in raw_names.items()}
    else:
        raise ValueError("names must be a list or mapping")

    if sorted(names) != list(range(len(names))) or not names:
        raise ValueError("Class IDs must be contiguous starting at zero")

    image_dirs: dict[str, Path] = {}
    for split in SPLITS:
        raw = cfg.get(split)
        if not isinstance(raw, str):
            raise ValueError(f"A directory path is required for {split}")
        folder = (root / raw).resolve()
        if not folder.is_relative_to(root):
            raise ValueError(f"{split} path escapes the dataset root: {raw}")
        image_dirs[split] = folder

    return Dataset(path, root, names, image_dirs)


def parse_label(path: Path, names: dict[int, str]) -> list[tuple[int, float, float, float, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing annotation: {path}")

    boxes = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue

        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path}:{line_no}: expected class + four coordinates")

        cls_f, x, y, w, h = map(float, values)
        if not all(math.isfinite(v) for v in (cls_f, x, y, w, h)):
            raise ValueError(f"{path}:{line_no}: non-finite label")
        if not cls_f.is_integer() or int(cls_f) not in names:
            raise ValueError(f"{path}:{line_no}: invalid class ID")
        if w <= 0 or h <= 0:
            raise ValueError(f"{path}:{line_no}: non-positive box size")
        if min(x - w / 2, y - h / 2) < -1e-6 or max(x + w / 2, y + h / 2) > 1 + 1e-6:
            raise ValueError(f"{path}:{line_no}: box outside normalized image bounds")

        boxes.append((int(cls_f), x, y, w, h))

    return boxes


def validate_dataset(dataset: Dataset) -> dict[str, Any]:
    image_names: set[str] = set()
    expected_labels: set[Path] = set()
    hashes: dict[str, list[str]] = defaultdict(list)
    summaries: dict[str, Any] = {}
    image_sizes: Counter[tuple[int, int]] = Counter()

    for split in SPLITS:
        class_counts: Counter[int] = Counter()
        negatives = 0
        images = dataset.images(split)

        for image in images:
            if image.name in image_names:
                raise ValueError(f"Repeated image filename across dataset: {image.name}")
            image_names.add(image.name)

            label = dataset.label_path(image)
            expected_labels.add(label.resolve())
            boxes = parse_label(label, dataset.names)

            class_counts.update(box[0] for box in boxes)
            negatives += int(not boxes)
            hashes[sha256_file(image)].append(f"{split}/{image.name}")

            with Image.open(image) as im:
                im.verify()
                image_sizes[im.size] += 1

        summaries[split] = {
            "images": len(images),
            "negative_images": negatives,
            "objects_by_class": {
                dataset.names[class_id]: class_counts[class_id]
                for class_id in dataset.names
            },
        }

    actual_labels = {
        p.resolve()
        for p in dataset.root.rglob("*.txt")
        if "labels" in p.relative_to(dataset.root).parts
    }
    extra_labels = sorted(
        str(p.relative_to(dataset.root))
        for p in actual_labels - expected_labels
    )

    return {
        "total_images": len(image_names),
        "classes": dataset.names,
        "splits": summaries,
        "image_sizes": {
            f"{width}x{height}": count
            for (width, height), count in sorted(image_sizes.items())
        },
        "extra_labels": extra_labels,
        "duplicate_image_groups": [
            files for files in hashes.values() if len(files) > 1
        ],
    }


def make_binary_dataset(source: Dataset, output: Path) -> Path:
    output = output.resolve()

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing dataset: {output}")
    if output.is_relative_to(source.root):
        raise ValueError("Binary output must be outside the source dataset")

    plan: list[tuple[str, Path, Path]] = []
    for split in SPLITS:
        for image in source.images(split):
            label = source.label_path(image)
            parse_label(label, source.names)
            plan.append((split, image, label))

    output.mkdir(parents=True)

    for split, image, label in plan:
        relative = image.relative_to(source.image_dirs[split])
        dst_image = output / "images" / split / relative
        dst_label = (output / "labels" / split / relative).with_suffix(".txt")

        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, dst_image)

        lines = []
        for line in label.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                parts = line.split()
                lines.append(" ".join(["0", *parts[1:]]))

        dst_label.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "defect"},
    }
    yaml_path = output / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return yaml_path
