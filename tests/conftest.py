from pathlib import Path
import pytest
import yaml
from PIL import Image


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for suffix, label in [("positive", "2 0.5 0.5 0.4 0.2\n"), ("negative", "")]:
            name = f"{split}_{suffix}"
            Image.new("RGB", (16, 16)).save(image_dir / f"{name}.png")
            (label_dir / f"{name}.txt").write_text(label)
    cfg = {"path": ".", "train": "images/train", "val": "images/val", "test": "images/test",
           "names": {0: "broken_wire", 1: "corrosion", 2: "kink_or_bend"}}
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path
