import csv
import json
import math
import os
import re
import tempfile

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

MASK_ALIGNMENT_POLICIES = ["source-frame", "full-frame", "flux2-crop"]
MASK_ALIGNMENT_HELP = (
    "how masks drawn on the source image map onto the edited image: source-frame if the edited image "
    "has the source resolution, full-frame if it shows the whole source frame at another resolution "
    "(MIRAGE, Qwen), flux2-crop for the official FLUX.2 pipelines, which downscale inputs to 1 MP and "
    "center-crop them to multiples of 16"
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_samples(annotations_jsonl, crop_instruction_jsonl):
    """(image name, instruction, masks, {edit index: local instruction}) for each benchmark image."""
    local = {}
    for record in load_jsonl(crop_instruction_jsonl):
        index = int(re.search(r"\d+", record["image"])[0]) - 1
        local.setdefault(record["original_image"], {})[index] = record["new_instruction"].strip()
    samples = []
    for record in load_jsonl(annotations_jsonl):
        name = record["image"]
        masks = record["mask"]
        if isinstance(masks, str):
            # RefEdit stores one edit's polygons as a JSON string. MIRA stores
            # the per-edit mask list directly, including for top-N subsets.
            try:
                masks = [json.loads(masks)]
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid mask JSON for {name} in {annotations_jsonl}") from error
        if not isinstance(masks, list) or not masks:
            raise ValueError(f"Missing or invalid masks for {name} in {annotations_jsonl}")
        for index, mask in enumerate(masks):
            if not isinstance(mask, list) or not mask:
                raise ValueError(f"Empty or invalid mask for {name}, edit {index}, in {annotations_jsonl}")
            polygons = mask if isinstance(mask[0], list) else [mask]
            if any(not isinstance(polygon, list) or not polygon for polygon in polygons):
                raise ValueError(f"Empty or invalid polygon for {name}, edit {index}, in {annotations_jsonl}")
        if "instruction_count" in record:
            count = record["instruction_count"]
            if type(count) is not int or count <= 0:
                raise ValueError(f"Invalid instruction_count for {name}: expected a positive integer")
            if len(masks) != count:
                raise ValueError(f"{name}: instruction_count is {count}, but found {len(masks)} masks")
            image_edits = local.get(name, {})
            missing = [index for index in range(count) if index not in image_edits]
            if missing:
                raise ValueError(f"Missing GT edit indices {missing} for {name} in {crop_instruction_jsonl}")
            edits = {index: image_edits[index] for index in range(count)}
        else:
            edits = dict(sorted(local[name].items()))
        samples.append((name, record["editing_instruction"], masks, edits))
    return samples


def select_evaluation_samples(args, samples):
    """Select available output pairs while retaining their annotation keys."""
    selected, skipped = [], []
    skip_missing = getattr(args, "skip_missing_edits", False)
    for key, sample in enumerate(samples):
        name = sample[0]
        source_path = os.path.join(args.input_image_root, name)
        edited_path = os.path.join(args.edited_image_root, name)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Missing original image: {source_path}")
        if not os.path.exists(edited_path):
            if skip_missing:
                skipped.append(name)
                continue
            raise FileNotFoundError(
                f"Missing edited image: {edited_path}. Use --skip-missing-edits to exclude missing outputs."
            )
        if not os.path.isfile(edited_path):
            raise ValueError(f"Edited image is not a file: {edited_path}")
        selected.append((key, sample))
    if not selected:
        raise ValueError("No image pairs available for evaluation")
    return selected, skipped


def polygon_mask(mask, size):
    polygons = mask if isinstance(mask[0], list) else [mask]
    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        draw.polygon(polygon, outline=1, fill=1)
    return np.array(canvas)


def project_mask(mask, size, policy):
    if policy == "source-frame":
        return mask
    if policy == "full-frame":
        return mask if mask.size == size else mask.resize(size, Image.Resampling.NEAREST)
    width, height = mask.size
    if width * height > 1024 * 1024:
        scale = math.sqrt(1024 * 1024 / (width * height))
        mask = mask.resize((int(width * scale), int(height * scale)), Image.Resampling.NEAREST)
    left = (mask.width - size[0]) // 2
    top = (mask.height - size[1]) // 2
    return mask.crop((left, top, left + size[0], top + size[1]))


def masked_pair(source, edited, mask, policy):
    source_mask = Image.fromarray(polygon_mask(mask, source.size))
    edited_mask = project_mask(source_mask, edited.size, policy)
    return (
        Image.fromarray(np.array(source) * np.array(source_mask)[:, :, None]),
        Image.fromarray(np.array(edited) * np.array(edited_mask)[:, :, None]),
    )


def _mean(rows, key):
    values = [row[key] for row in rows if key in row]
    return sum(values) / len(values) if values else None


def _overall(metrics):
    return math.sqrt(min(metrics["prompt_following"], metrics["consistency"]) * metrics["perceptual_quality"])


def _read_resume_jsonl(path):
    if not os.path.exists(path):
        return [], False
    with open(path, "rb") as f:
        data = f.read()
    lines = [line for line in data.splitlines() if line.strip()]
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if index != len(lines) - 1:
                raise ValueError(f"Cannot resume: corrupt interior JSONL line in {path}") from error
            return rows, True
    return rows, bool(data and not data.endswith(b"\n"))


def _resume_results(edit_path, image_path, samples, score_pq, score_sc):
    """Validate both logs before repairing interrupted writes; image rows are commit markers."""
    edit_rows, repair_edits = _read_resume_jsonl(edit_path)
    image_rows, repair_images = _read_resume_jsonl(image_path)
    expected = {str(key): sample for key, sample in enumerate(samples)}
    committed, stored_edits = {}, {}

    def invalid(message):
        raise ValueError("Cannot resume: " + message)

    def check_score(row, field):
        value = row.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 10:
            invalid(f"invalid {field} in result {row.get('key')!r}")

    for row in image_rows:
        if not isinstance(row, dict):
            invalid("per-image record must be an object")
        key = row.get("key")
        if not isinstance(key, str) or key not in expected or key in committed:
            invalid(f"unknown or duplicate image key {key!r}")
        name, _, _, edits = expected[key]
        if row.get("image") != name or type(row.get("num_crops")) is not int or row["num_crops"] != len(edits):
            invalid(f"image or crop count mismatch for key {key}")
        for field, required in (
            ("perceptual_quality", score_pq), ("PQ_reasoning", score_pq),
            ("prompt_following", score_sc), ("consistency", score_sc), ("SC_reasoning", score_sc),
            ("overall", score_pq and score_sc),
        ):
            if (field in row) != bool(required):
                invalid(f"metric fields mismatch for image {key}: {field}")
        if score_pq:
            check_score(row, "perceptual_quality")
            if not isinstance(row["PQ_reasoning"], str):
                invalid(f"invalid PQ reasoning for image {key}")
        if score_sc:
            for field in ("prompt_following", "consistency"):
                check_score(row, field)
        if score_pq and score_sc:
            check_score(row, "overall")
            if not math.isclose(row["overall"], _overall(row)):
                invalid(f"inconsistent overall score for image {key}")
        committed[key] = row

    for row in edit_rows:
        if not isinstance(row, dict):
            invalid("per-edit record must be an object")
        key, index = row.get("image_key"), row.get("crop_index")
        if not isinstance(key, str) or key not in expected or type(index) is not int:
            invalid(f"invalid image key or crop index in edit {row.get('key')!r}")
        edit_key = f"{key}__{index}"
        if not score_sc or index not in expected[key][3] or row.get("key") != edit_key or edit_key in stored_edits:
            invalid(f"unknown, mismatched or duplicate edit key {row.get('key')!r}")
        stored_edits[edit_key] = row
        if key in committed:
            for field in ("prompt_following", "consistency"):
                check_score(row, field)
            if not isinstance(row.get("SC_reasoning"), str):
                invalid(f"invalid SC reasoning for edit {edit_key}")

    if score_sc:
        for key, image_row in committed.items():
            crop_keys = [f"{key}__{index}" for index in expected[key][3]]
            if not crop_keys or any(crop_key not in stored_edits for crop_key in crop_keys):
                invalid(f"missing committed crop results for image {key}")
            crops = [stored_edits[crop_key] for crop_key in crop_keys]
            for field in ("prompt_following", "consistency"):
                if not math.isclose(image_row[field], _mean(crops, field)):
                    invalid(f"inconsistent {field} for image {key}")
            if image_row["SC_reasoning"] != [row["SC_reasoning"] for row in crops]:
                invalid(f"inconsistent SC reasoning for image {key}")

    completed_edits = [row for row in edit_rows if row["image_key"] in committed]
    for path, rows, repair in (
        (edit_path, completed_edits, repair_edits or len(completed_edits) != len(edit_rows)),
        (image_path, image_rows, repair_images),
    ):
        if repair:
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=os.path.dirname(path), prefix=".resume-", delete=False
                ) as f:
                    temporary_path = f.name
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                os.replace(temporary_path, path)
            finally:
                if temporary_path is not None and os.path.exists(temporary_path):
                    os.unlink(temporary_path)
    return completed_edits, image_rows


def evaluate(args, score_pq=None, score_sc=None):
    """Score all edited images and write per-edit, per-image and summary results.

    score_pq(source, edited) -> (score, reasoning)
    score_sc([(index, instruction, masked source, masked edited)]) -> {index: (PF, Cons, reasoning)}
    """
    samples = load_samples(args.annotations_jsonl, args.crop_instruction_jsonl)
    selected, skipped_images = select_evaluation_samples(args, samples)
    if skipped_images:
        tqdm.write(f"Skipping {len(skipped_images)} missing edited images: {', '.join(skipped_images)}")
    os.makedirs(args.result_dir, exist_ok=True)
    edit_path = os.path.join(args.result_dir, "per_edit_results.jsonl")
    image_path = os.path.join(args.result_dir, "per_image_results.jsonl")
    resume = getattr(args, "resume", False)
    edit_rows, image_rows = [], []
    if resume:
        if skipped_images:
            saved_rows, _ = _read_resume_jsonl(image_path)
            skipped = set(skipped_images)
            committed_skipped = [
                row["image"] for row in saved_rows
                if isinstance(row, dict) and row.get("image") in skipped
            ]
            if committed_skipped:
                raise ValueError(
                    "Cannot resume: previously scored edited images are now missing: "
                    + ", ".join(committed_skipped)
                    + ". Restore the outputs or use a new result directory."
                )
        edit_rows, image_rows = _resume_results(edit_path, image_path, samples, score_pq, score_sc)
        tqdm.write(f"Resuming: {len(image_rows)}/{len(selected)} images already completed")
    completed = {row["key"] for row in image_rows}
    with (
        open(edit_path, "a" if resume else "w", encoding="utf-8") as edit_file,
        open(image_path, "a" if resume else "w", encoding="utf-8") as image_file,
    ):
        for key, (name, _, masks, edits) in tqdm(selected):
            if str(key) in completed:
                continue
            source = Image.open(os.path.join(args.input_image_root, name)).convert("RGB")
            edited = Image.open(os.path.join(args.edited_image_root, name)).convert("RGB")
            row = {"key": str(key), "image": name, "num_crops": len(edits)}

            if score_pq:
                row["perceptual_quality"], row["PQ_reasoning"] = score_pq(source, edited)
            if score_sc:
                items = [
                    (i, text, *masked_pair(source, edited, masks[i], args.mask_alignment_policy))
                    for i, text in edits.items()
                ]
                scores = score_sc(items)
                rows = []
                for i in edits:
                    prompt_following, consistency, reasoning = scores[i]
                    rows.append(
                        {
                            "key": f"{key}__{i}",
                            "image_key": str(key),
                            "crop_index": i,
                            "prompt_following": prompt_following,
                            "consistency": consistency,
                            "SC_reasoning": reasoning,
                        }
                    )
                    edit_file.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
                edit_file.flush()
                edit_rows += rows
                row["prompt_following"] = _mean(rows, "prompt_following")
                row["consistency"] = _mean(rows, "consistency")
                row["SC_reasoning"] = [r["SC_reasoning"] for r in rows]
            if score_pq and score_sc:
                row["overall"] = _overall(row)

            # This image record commits the already-flushed per-edit records for resume.
            image_rows.append(row)
            image_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            image_file.flush()

    perceptual_quality = _mean(image_rows, "perceptual_quality")
    summary = {"num_images": len(image_rows), "num_edits": len(edit_rows)}
    if getattr(args, "skip_missing_edits", False):
        summary.update({
            "num_input_images": len(samples),
            "num_skipped_images": len(skipped_images),
            "skipped_images": skipped_images,
        })
    for granularity, rows in (("per_edit", edit_rows), ("per_image", image_rows)):
        metrics = {
            "prompt_following": _mean(rows, "prompt_following"),
            "consistency": _mean(rows, "consistency"),
            "perceptual_quality": perceptual_quality if rows else None,
        }
        metrics["overall"] = None if None in metrics.values() else _overall(metrics)
        rounded = {k: None if v is None else round(v, 4) for k, v in metrics.items()}
        summary[granularity] = {"count": len(rows), **rounded}

    with open(os.path.join(args.result_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.result_dir, "summary.csv"), "w", encoding="utf-8", newline="") as f:
        fields = ["granularity", "count", "prompt_following", "consistency", "perceptual_quality", "overall"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for granularity in ("per_edit", "per_image"):
            writer.writerow({"granularity": granularity, **summary[granularity]})
    print(json.dumps(summary, indent=2))
