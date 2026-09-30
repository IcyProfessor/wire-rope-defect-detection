"""Command-line interface for the project."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .common import new_output_directory, read_csv, write_csv, write_json
from .dataset import load_dataset, make_binary_dataset, validate_dataset
from .decision import decide, load_decision_config, simulate_journal
from .reporting import journal_summary


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Wire-rope defect detection experiment utilities"
    )
    sub = command.add_subparsers(dest="command", required=True)

    item = sub.add_parser("validate-data", help="Validate YOLO images and annotations")
    item.add_argument("--data", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)

    item = sub.add_parser(
        "prepare-binary",
        help="Copy the dataset and merge all defect class IDs into class 0",
    )
    item.add_argument("--data", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)

    item = sub.add_parser("summarize-journal", help="Calculate image-level journal metrics")
    item.add_argument(
        "--journal",
        type=Path,
        default=Path("artifacts/decision/diagnostic_journal.csv"),
    )
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--config", type=Path, default=None)

    item = sub.add_parser("decide", help="Apply decision rules to camera and Hall scores")
    item.add_argument("--camera", type=float, required=True)
    item.add_argument("--hall", type=float, required=True)
    item.add_argument("--config", type=Path, default=None)

    item = sub.add_parser("train", help="Run YOLOv8s transfer learning")
    item.add_argument("--data", type=Path, required=True)
    item.add_argument("--config", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--device", default="cpu")
    item.add_argument("--initial-weights", default=None)
    item.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the training request without starting YOLO",
    )

    item = sub.add_parser("predict", help="Run YOLO inference")
    item.add_argument("--weights", type=Path, required=True)
    item.add_argument("--source", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--device", default="cpu")
    item.add_argument("--conf", type=float, default=0.05)
    item.add_argument("--iou", type=float, default=0.5)

    item = sub.add_parser(
        "evaluate",
        help="Calculate object-detection and fixed-threshold image metrics",
    )
    item.add_argument("--weights", type=Path, required=True)
    item.add_argument("--data", type=Path, required=True)
    item.add_argument("--split", choices=("val", "test"), default="test")
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--device", default="cpu")
    item.add_argument("--threshold", type=float, default=0.35)

    item = sub.add_parser(
        "simulate",
        help="Generate the demonstration Hall/position journal from camera predictions",
    )
    item.add_argument("--predictions", type=Path, required=True)
    item.add_argument("--data", type=Path, required=True)
    item.add_argument("--split", choices=("val", "test"), default="test")
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--seed", type=int, default=42)
    item.add_argument("--config", type=Path, default=None)

    return command


def main(argv: list[str] | None = None) -> None:
    command = parser()
    args = command.parse_args(argv)

    try:
        if args.command == "validate-data":
            report = validate_dataset(load_dataset(args.data))
            if args.output.exists():
                raise FileExistsError(f"Refusing to overwrite {args.output}")
            write_json(args.output, report)
            print(f"Validated {report['total_images']} images. Report: {args.output}")

        elif args.command == "prepare-binary":
            print(make_binary_dataset(load_dataset(args.data), args.output))

        elif args.command == "summarize-journal":
            rows = read_csv(args.journal)
            cfg = load_decision_config(args.config)
            output = new_output_directory(args.output)
            write_json(
                output / "image_level_summary.json",
                journal_summary(rows, cfg.visual_threshold),
            )
            print(f"Saved journal summary: {output}")

        elif args.command == "decide":
            result = decide(
                args.camera,
                args.hall,
                load_decision_config(args.config),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))

        elif args.command == "train":
            from .yolo import train, training_plan

            if args.dry_run:
                print(json.dumps(
                    training_plan(
                        args.data,
                        args.config,
                        args.output,
                        args.device,
                        args.initial_weights,
                    ),
                    indent=2,
                ))
            else:
                print(train(
                    args.data,
                    args.config,
                    args.output,
                    args.device,
                    args.initial_weights,
                ))

        elif args.command == "predict":
            from .yolo import predict

            print(predict(
                args.weights,
                args.source,
                args.output,
                args.device,
                args.conf,
                args.iou,
            ))

        elif args.command == "evaluate":
            from .yolo import evaluate

            print(evaluate(
                args.weights,
                args.data,
                args.split,
                args.output,
                args.device,
                args.threshold,
            ))

        elif args.command == "simulate":
            from .yolo import camera_rows_with_ground_truth

            cfg = load_decision_config(args.config)
            camera_rows = camera_rows_with_ground_truth(
                args.predictions,
                args.data,
                args.split,
                cfg.visual_threshold,
            )
            journal = simulate_journal(camera_rows, args.seed, cfg)

            output = new_output_directory(args.output)
            write_csv(output / "simulated_journal.csv", journal)
            write_json(
                output / "image_level_summary.json",
                journal_summary(journal, cfg.visual_threshold),
            )
            write_json(
                output / "simulation_config.json",
                {"seed": args.seed, **asdict(cfg)},
            )
            print(f"Saved simulated journal: {output}")

    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        command.exit(2, f"Error: {exc}\n")
