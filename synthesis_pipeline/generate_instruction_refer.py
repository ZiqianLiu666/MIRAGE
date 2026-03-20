import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.qwen_utils as qwen

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Three-stage pipeline with warning-only validation, one retry for generation/extraction, and feedback-on-failure. Real-time saving."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--slot-template", required=True)
    parser.add_argument("--generator-template", required=True)
    parser.add_argument("--extractor-template", required=True)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--retries", type=int, default=1, help="Additional retries after the first attempt. Default 1 => up to 2 attempts total.")
    parser.add_argument("--slot-max-new-tokens", type=int, default=384)
    parser.add_argument("--gen-max-new-tokens", type=int, default=896)
    parser.add_argument("--ext-max-new-tokens", type=int, default=384)
    return parser.parse_args()


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    items = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: List[Dict[str, Any]]):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp, out)


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")
    
def extract_json_any(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def parse_slot_plan(text: str) -> Optional[Dict[str, Any]]:
    obj = extract_json_any(text)
    if not isinstance(obj, dict):
        return None
    n = obj.get("repeated_count")
    cat = obj.get("repeated_category")
    repeated_slots = obj.get("repeated_slots")
    repeated_details = obj.get("repeated_details")
    preferred_non_repeated_targets = obj.get("preferred_non_repeated_targets", [])
    if not isinstance(n, int) or n not in (3, 4, 5):
        return None
    if not isinstance(cat, str) or not cat.strip():
        return None
    if not isinstance(repeated_slots, list) or len(repeated_slots) != n:
        return None
    if not isinstance(repeated_details, list) or len(repeated_details) != n:
        return None
    rs = []
    for x in repeated_slots:
        if not isinstance(x, str) or not x.strip():
            return None
        rs.append(x.strip())
    if len(set(rs)) != len(rs):
        return None
    rd = []
    for x in repeated_details:
        if not isinstance(x, str):
            return None
        rd.append(x.strip())
    pnrt = []
    if not isinstance(preferred_non_repeated_targets, list):
        return None
    for x in preferred_non_repeated_targets:
        if not isinstance(x, str) or not x.strip():
            return None
        pnrt.append(x.strip())
    if len(set(pnrt)) != len(pnrt):
        return None
    return {
        "repeated_count": n,
        "repeated_category": cat.strip(),
        "repeated_slots": rs,
        "repeated_details": rd,
        "preferred_non_repeated_targets": pnrt,
    }


def parse_slot_instructions(text: str) -> Optional[List[Dict[str, str]]]:
    obj = extract_json_any(text)
    if not isinstance(obj, dict):
        return None
    arr = obj.get("slot_instructions")
    if not isinstance(arr, list) or len(arr) != 5:
        return None
    out = []
    for i, row in enumerate(arr, start=1):
        if not isinstance(row, dict):
            return None
        if row.get("slot_id") != i:
            return None
        target = row.get("target")
        ins = row.get("edit_instruction")
        if not isinstance(target, str) or not target.strip():
            return None
        if not isinstance(ins, str) or not ins.strip():
            return None
        out.append({"slot_id": i, "target": target.strip(), "edit_instruction": ins.strip()})
    return out


def parse_refer_objects(text: str) -> Optional[List[str]]:
    obj = extract_json_any(text)
    arr = None
    if isinstance(obj, dict):
        arr = obj.get("refer_object")
        if arr is None:
            arr = obj.get("Refer_object")
    elif isinstance(obj, list):
        arr = obj
    if not isinstance(arr, list) or len(arr) != 5:
        return None
    out = []
    for x in arr:
        if not isinstance(x, str) or not x.strip():
            return None
        out.append(x.strip())
    return out


def warn_stage1(plan: Dict[str, Any], slot_instructions: List[Dict[str, str]]) -> List[str]:
    warnings = []
    n = plan["repeated_count"]
    repeated_slots = plan["repeated_slots"]
    if len(slot_instructions) != 5:
        warnings.append("wrong number of slot_instructions")
        return warnings
    for i in range(min(n, len(slot_instructions))):
        expected_target = repeated_slots[i]
        row = slot_instructions[i]
        if row["target"] != expected_target:
            warnings.append(f"slot_instructions[{i}] target mismatch: expected {expected_target}")
        if expected_target not in row["edit_instruction"]:
            warnings.append(f"slot_instructions[{i}] edit_instruction does not contain target verbatim: {expected_target}")
    return warnings


def warn_stage2(instructions: List[str], refer_objects: Optional[List[str]]) -> List[str]:
    warnings = []
    if refer_objects is None:
        warnings.append("could not parse 5 refer_object")
        return warnings
    if len(refer_objects) != 5:
        warnings.append("wrong number of refer_object")
        return warnings
    if len(set(refer_objects)) != 5:
        warnings.append("refer_object contains duplicates")
    for i, (ins, ref) in enumerate(zip(instructions, refer_objects)):
        if ref not in ins:
            warnings.append(f"refer_object[{i}] is not exact substring of instruction[{i}]")
    return warnings


def build_feedback(reasons: List[str]) -> str:
    if not reasons:
        return "- None."
    uniq = []
    seen = set()
    for r in reasons:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return "\n".join(f"- {r}" for r in uniq[-6:])


def build_slot_messages(template: str, source_prompt: str):
    text = template.replace("{source_prompt}", source_prompt)
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def build_generator_messages(template: str, image: Image.Image, image_name: str, slot_plan: Dict[str, Any], failure_feedback: str):
    text = template.replace("{slot_plan_json}", json.dumps(slot_plan, ensure_ascii=False, indent=2))
    text = text.replace("{failure_feedback}", failure_feedback)
    text += f"\n\nIMAGE_FILENAME: {image_name}\n"
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]


def build_extractor_messages(template: str, instruction_list: List[str], failure_feedback: str):
    text = template.replace("{instruction_list_json}", json.dumps(instruction_list, ensure_ascii=False))
    text = text.replace("{failure_feedback}", failure_feedback)
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def combine_instructions(instructions: List[str]) -> str:
    parts = []
    for s in instructions:
        s = s.strip()
        if s.endswith("."):
            s = s[:-1]
        parts.append(s)
    out = ", and ".join(parts)
    if not out.endswith("."):
        out += "."
    return out


def ordered_output(source_records: List[Dict[str, Any]], result_by_image: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered, seen = [], set()
    for rec in source_records:
        image_name = rec.get("image")
        if isinstance(image_name, str) and image_name in result_by_image and image_name not in seen:
            ordered.append(result_by_image[image_name]); seen.add(image_name)
    for image_name, rec in result_by_image.items():
        if image_name not in seen:
            ordered.append(rec)
    return ordered


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)

    slot_template = load_text(args.slot_template)
    gen_template = load_text(args.generator_template)
    ext_template = load_text(args.extractor_template)

    source_records = load_jsonl(args.jsonl)
    if args.max_items is not None:
        source_records = source_records[:args.max_items]

    existing = load_jsonl(args.out_jsonl) if Path(args.out_jsonl).exists() else []
    result_by_image = {item["image"]: item for item in existing if isinstance(item.get("image"), str)}

    tasks = []
    for rec in source_records:
        image_name = rec.get("image")
        if not isinstance(image_name, str) or not image_name:
            continue
        if image_name in result_by_image:
            continue
        img_path = image_dir / image_name
        if not img_path.exists():
            print(f"[WARN] missing image: {img_path}")
            continue
        tasks.append((image_name, img_path, rec.get("source_prompt", "")))

    if not tasks:
        print("No records to process.")
        return

    total_attempts = args.retries + 1

    for image_name, img_path, source_prompt in tqdm(tasks, desc="slot+gen+extract"):
        image = load_image(img_path)

        slot_out = (qwen.get_backend().chat_batch([build_slot_messages(slot_template, source_prompt)], max_new_tokens=args.slot_max_new_tokens) or [""])[0]
        slot_plan = parse_slot_plan(slot_out)

        last_slot_instructions = None
        gen_warnings: List[str] = []

        for attempt in range(1, total_attempts + 1):
            gen_out = (qwen.get_backend().chat_batch([build_generator_messages(gen_template, image, image_name, slot_plan, build_feedback(gen_warnings))], max_new_tokens=args.gen_max_new_tokens) or [""])[0]
            slot_instructions = parse_slot_instructions(gen_out)
            if slot_instructions is None:
                reason = "could not parse 5 slot_instructions"
                print(f"[GEN FAIL] {image_name} try {attempt}: {reason}")
                gen_warnings.append(reason)
                continue
            last_slot_instructions = slot_instructions
            curr_warnings = warn_stage1(slot_plan, slot_instructions)
            if curr_warnings and attempt < total_attempts:
                for w in curr_warnings:
                    print(f"[PLAN FAIL] {image_name} try {attempt}: {w}")
                gen_warnings.extend(curr_warnings)
                continue
            for w in curr_warnings:
                print(f"[PLAN FAIL] {image_name}: {w}")
            break

        if last_slot_instructions is None:
            write_jsonl(args.out_jsonl, ordered_output(source_records, result_by_image))
            continue

        instructions = [row["edit_instruction"] for row in last_slot_instructions]

        last_refer_objects = None
        ext_warnings: List[str] = []

        for attempt in range(1, total_attempts + 1):
            ext_out = (qwen.get_backend().chat_batch([build_extractor_messages(ext_template, instructions, build_feedback(ext_warnings))], max_new_tokens=args.ext_max_new_tokens) or [""])[0]
            refer_objects = parse_refer_objects(ext_out)
            if refer_objects is None:
                reason = "could not parse 5 refer_object"
                print(f"[EXT FAIL] {image_name} try {attempt}: {reason}")
                ext_warnings.append(reason)
                continue
            last_refer_objects = refer_objects
            curr_warnings = warn_stage2(instructions, refer_objects)
            if curr_warnings and attempt < total_attempts:
                for w in curr_warnings:
                    print(f"[VALIDATION FAIL] {image_name} try {attempt}: {w}")
                ext_warnings.extend(curr_warnings)
                continue
            for w in curr_warnings:
                print(f"[VALIDATION FAIL] {image_name}: {w}")
            break

        result_by_image[image_name] = {
            "image": image_name,
            "editing_instruction": combine_instructions(instructions),
            "refer_object": last_refer_objects if last_refer_objects is not None else [],
        }
        write_jsonl(args.out_jsonl, ordered_output(source_records, result_by_image))

    print(f"Done. wrote {len(result_by_image)} rows to {args.out_jsonl}")


if __name__ == "__main__":
    main()
