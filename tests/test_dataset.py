from pathlib import Path

import pytest

from rope_defect.common import sha256_file
from rope_defect.dataset import (
    load_dataset,
    make_binary_dataset,
    parse_label,
    validate_dataset,
)


def test_valid_dataset(tiny_dataset):
    report = validate_dataset(load_dataset(tiny_dataset))
    assert report["total_images"] == 6
    assert report["splits"]["train"]["negative_images"] == 1
    assert report["splits"]["train"]["objects_by_class"]["kink_or_bend"] == 1
    assert report["duplicate_image_groups"]


def test_conversion_preserves_images_and_coordinates(tiny_dataset, tmp_path):
    source = load_dataset(tiny_dataset)
    binary = load_dataset(make_binary_dataset(source, tmp_path / "binary"))

    assert binary.names == {0: "defect"}

    for split in ("train", "val", "test"):
        before = source.images(split)
        after = binary.images(split)
        assert [path.name for path in before] == [path.name for path in after]

        for source_image, binary_image in zip(before, after):
            assert sha256_file(source_image) == sha256_file(binary_image)

            old_boxes = parse_label(
                source.label_path(source_image),
                source.names,
            )
            new_boxes = parse_label(
                binary.label_path(binary_image),
                binary.names,
            )

            assert [box[1:] for box in old_boxes] == [box[1:] for box in new_boxes]
            assert all(box[0] == 0 for box in new_boxes)


def test_no_inplace_or_existing_output(tiny_dataset, tmp_path):
    source = load_dataset(tiny_dataset)

    with pytest.raises(FileExistsError):
        make_binary_dataset(source, source.root)

    with pytest.raises(ValueError):
        make_binary_dataset(source, source.root / "nested")

    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        make_binary_dataset(source, existing)


@pytest.mark.parametrize(
    "text",
    [
        "0 0.5 0.5 0.2",
        "9 0.5 0.5 0.2 0.2",
        "0 nan 0.5 0.2 0.2",
        "0 0.9 0.5 0.5 0.2",
        "0 0.5 0.5 -0.1 0.2",
        "0.5 0.5 0.5 0.1 0.1",
    ],
)
def test_bad_label_is_not_silently_dropped(tmp_path, text):
    label = tmp_path / "bad.txt"
    label.write_text(text)

    with pytest.raises(ValueError):
        parse_label(label, {0: "defect"})


def test_missing_label_is_not_negative(tiny_dataset):
    source = load_dataset(tiny_dataset)
    source.label_path(source.images("test")[0]).unlink()

    with pytest.raises(FileNotFoundError):
        validate_dataset(source)


def test_yaml_is_relative_to_itself(tiny_dataset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    assert load_dataset(tiny_dataset).root == tiny_dataset.parent
