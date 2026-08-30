from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SC_IMAGE_FIELDS = {
    "num_sc_evals",
    "prompt_following",
    "consistency",
    "SC_reasoning",
    "sc_model",
    "overall",
}
SC_COPY_FIELDS = (
    "num_sc_evals",
    "prompt_following",
    "consistency",
    "SC_reasoning",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge GPT-5.1 PQ rows with GPT-5.6 Luna SC rows."
    )
    parser.add_argument("--pq-root", type=Path, required=True)
    parser.add_argument("--sc-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pq-model", default="gpt-5.1")
    parser.add_argument("--sc-model", default="gpt-5.6-luna")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def keyed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["key"]): row for row in rows}


def has_pq(row: dict[str, Any]) -> bool:
    return all(
        row.get(field) is not None
        for field in ("perceptual_quality", "PQ_reasoning")
    )


def has_sc(row: dict[str, Any]) -> bool:
    return all(
        row.get(field) is not None
        for field in ("prompt_following", "consistency", "SC_reasoning")
    )


def metric_mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def aggregate_metrics(
    sc_rows: list[dict[str, Any]],
    pq_rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    prompt_following = (
        metric_mean(sc_rows, "prompt_following") if sc_rows else None
    )
    consistency = metric_mean(sc_rows, "consistency") if sc_rows else None
    perceptual_quality = (
        metric_mean(pq_rows, "perceptual_quality") if pq_rows else None
    )
    overall = None
    if (
        prompt_following is not None
        and consistency is not None
        and perceptual_quality is not None
    ):
        overall = math.sqrt(
            min(prompt_following, consistency) * perceptual_quality
        )
    return {
        "prompt_following": prompt_following,
        "consistency": consistency,
        "perceptual_quality": perceptual_quality,
        "overall": overall,
    }


def rounded(metrics: dict[str, float | None]) -> dict[str, float | None]:
    return {
        key: None if value is None else round(value, 4)
        for key, value in metrics.items()
    }


def merge_model(
    model_name: str,
    pq_dir: Path,
    sc_dir: Path,
    output_dir: Path,
    pq_model: str,
    sc_model: str,
) -> dict[str, Any] | None:
    all_pq_rows = load_jsonl(pq_dir / "per_image_results.jsonl")
    pq_rows = [row for row in all_pq_rows if has_pq(row)]
    if not pq_rows:
        return None

    pq_by_key = keyed(pq_rows)
    sc_rows = [
        row
        for row in load_jsonl(sc_dir / "per_image_results.jsonl")
        if has_sc(row) and str(row["key"]) in pq_by_key
    ]
    sc_by_key = keyed(sc_rows)

    merged_rows: list[dict[str, Any]] = []
    for pq_row in pq_rows:
        key = str(pq_row["key"])
        merged = {
            field: value
            for field, value in pq_row.items()
            if field not in SC_IMAGE_FIELDS
        }
        merged["pq_model"] = pq_model

        sc_row = sc_by_key.get(key)
        if sc_row is not None:
            for field in SC_COPY_FIELDS:
                if field in sc_row:
                    merged[field] = sc_row[field]
            merged["sc_model"] = sc_model
            merged["overall"] = math.sqrt(
                min(
                    float(merged["prompt_following"]),
                    float(merged["consistency"]),
                )
                * float(merged["perceptual_quality"])
            )
        merged_rows.append(merged)

    sc_keys = set(sc_by_key)
    merged_edit_rows = []
    for row in load_jsonl(sc_dir / "per_edit_results.jsonl"):
        if str(row.get("image_key")) not in sc_keys:
            continue
        merged_edit_rows.append({**row, "sc_model": sc_model})

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_image_results.jsonl", merged_rows)
    write_jsonl(output_dir / "per_edit_results.jsonl", merged_edit_rows)

    checkpoint_path = pq_dir / "pq_checkpoints.jsonl"
    if checkpoint_path.exists():
        checkpoint_rows = [
            {**row, "pq_model": pq_model}
            for row in load_jsonl(checkpoint_path)
            if str(row.get("key")) in pq_by_key
        ]
        write_jsonl(output_dir / "pq_checkpoints.jsonl", checkpoint_rows)

    merged_sc_rows = [row for row in merged_rows if has_sc(row)]
    image_metrics = aggregate_metrics(merged_sc_rows, merged_rows)
    if merged_edit_rows:
        edit_metrics = aggregate_metrics(merged_edit_rows, merged_rows)
    else:
        edit_metrics = {
            "prompt_following": None,
            "consistency": None,
            "perceptual_quality": None,
            "overall": None,
        }

    summary = {
        "dataset_type": "mybench",
        "metrics": "all" if len(merged_sc_rows) == len(merged_rows) else "pq",
        "mask_alignment_policy": merged_rows[0].get("mask_alignment_policy"),
        "image_encoding": "png",
        "image_detail": "high",
        "models": {
            "sc": sc_model if merged_sc_rows else None,
            "pq": pq_model,
        },
        "api_requests": {"sc": 0, "pq": 0, "total": 0},
        "merge": {
            "pq_source": str(pq_dir),
            "sc_source": str(sc_dir) if sc_dir.is_dir() else None,
            "pq_images": len(merged_rows),
            "sc_images": len(merged_sc_rows),
        },
        "num_images": len(merged_rows),
        "num_edits": len(merged_edit_rows),
        "per_edit": {
            "count": len(merged_edit_rows),
            **rounded(edit_metrics),
        },
        "per_image": {
            "count": len(merged_rows),
            **rounded(image_metrics),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "granularity",
                "count",
                "prompt_following",
                "consistency",
                "perceptual_quality",
                "overall",
            ),
        )
        writer.writeheader()
        writer.writerow({"granularity": "per_edit", **summary["per_edit"]})
        writer.writerow({"granularity": "per_image", **summary["per_image"]})

    return {
        "model": model_name,
        "pq_images": len(merged_rows),
        "sc_images": len(merged_sc_rows),
        "edits": len(merged_edit_rows),
        "mode": (
            "SC(Luna)+PQ(GPT-5.1)"
            if merged_sc_rows
            else "PQ(GPT-5.1)-only"
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    merged: list[dict[str, Any]] = []
    skipped_no_pq: list[str] = []
    for pq_dir in sorted(path for path in args.pq_root.iterdir() if path.is_dir()):
        result = merge_model(
            model_name=pq_dir.name,
            pq_dir=pq_dir,
            sc_dir=args.sc_root / pq_dir.name,
            output_dir=args.output_root / pq_dir.name,
            pq_model=args.pq_model,
            sc_model=args.sc_model,
        )
        if result is None:
            skipped_no_pq.append(pq_dir.name)
        else:
            merged.append(result)

    manifest = {
        "pq_root": str(args.pq_root),
        "sc_root": str(args.sc_root),
        "output_root": str(args.output_root),
        "pq_model": args.pq_model,
        "sc_model": args.sc_model,
        "merged": merged,
        "skipped_without_pq": skipped_no_pq,
    }
    (args.output_root / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
