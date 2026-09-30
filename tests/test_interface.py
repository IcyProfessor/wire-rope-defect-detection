import json
from pathlib import Path
from types import SimpleNamespace

from rope_defect.cli import main
from rope_defect.yolo import collect_predictions, training_plan

ROOT = Path(__file__).resolve().parents[1]


def test_training_plan_preserves_parameters(tiny_dataset, tmp_path):
    plan = training_plan(
        tiny_dataset,
        ROOT / "configs/multiclass.yaml",
        tmp_path / "run",
    )
    assert plan["initial_model"] == "yolov8s.pt"
    assert plan["kwargs"]["resume"] is False
    assert plan["kwargs"]["patience"] == 40
    assert plan["kwargs"]["mosaic"] == 0.25
    assert plan["kwargs"]["epochs"] == 150


def test_dry_run_does_not_import_yolo(tiny_dataset, tmp_path, capsys):
    main([
        "train",
        "--data",
        str(tiny_dataset),
        "--config",
        str(ROOT / "configs/multiclass.yaml"),
        "--output",
        str(tmp_path / "run"),
        "--dry-run",
    ])

    assert json.loads(capsys.readouterr().out)["num_classes"] == 3
    assert not (tmp_path / "run").exists()


def test_validate_data_cli(tiny_dataset, tmp_path):
    output = tmp_path / "report.json"
    main([
        "validate-data",
        "--data",
        str(tiny_dataset),
        "--output",
        str(output),
    ])

    report = json.loads(output.read_text())
    assert report["total_images"] == 6


def test_journal_summary_cli(tmp_path):
    output = tmp_path / "summary"
    main([
        "summarize-journal",
        "--journal",
        str(ROOT / "artifacts/decision/diagnostic_journal.csv"),
        "--output",
        str(output),
    ])

    report = json.loads((output / "image_level_summary.json").read_text())
    assert report["vision_only"]["tp"] == 11
    assert report["vision_only"]["fn"] == 1


def test_prediction_adapter_contract_without_real_yolo(tmp_path):
    class Values:
        def __init__(self, values):
            self.values = values

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class FakeModel:
        def predict(self, **kwargs):
            assert "data" not in kwargs
            assert "labels" not in kwargs
            assert kwargs["conf"] == 0.05
            assert kwargs["iou"] == 0.5

            return [
                SimpleNamespace(
                    path="image.jpg",
                    boxes=SimpleNamespace(
                        xyxy=Values([[1, 2, 3, 4]]),
                        conf=Values([0.123456]),
                        cls=Values([0]),
                    ),
                )
            ]

    rows = collect_predictions(
        FakeModel(),
        [Path("image.jpg")],
        tmp_path,
        "cpu",
    )

    assert rows[0]["camera_conf"] == 0.1235
    assert rows[0]["camera_conf_raw"] == 0.123456
    assert rows[0]["boxes"][0]["xyxy"] == [1.0, 2.0, 3.0, 4.0]
