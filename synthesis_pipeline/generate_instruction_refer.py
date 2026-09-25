import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirage.vlm import BACKENDS, load_vlm  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate five edit instructions and their referring expressions per image (Sec. A.3)."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--jsonl", required=True, help="source prompts from generate_source_prompts.py")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--slot-template", required=True)
    parser.add_argument("--generator-template", required=True)
    parser.add_argument("--extractor-template", required=True)
    parser.add_argument("--vlm", default="qwen8b", choices=BACKENDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--retries", type=int, default=1, help="extra attempts, with feedback, after a failure")
    parser.add_argument("--slot-max-new-tokens", type=int, default=384)
    parser.add_argument("--gen-max-new-tokens", type=int, default=896)
    parser.add_argument("--ext-max-new-tokens", type=int, default=384)
    return parser.parse_args()


def load_json(text):
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def parse_slot_instructions(text):
    data = load_json(text)
    rows = data.get("slot_instructions") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) != 5:
        return None
    parsed = []
    for slot_id, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or row.get("slot_id") != slot_id:
            return None
        target, instruction = row.get("target"), row.get("edit_instruction")
        if not (isinstance(target, str) and target.strip() and isinstance(instruction, str) and instruction.strip()):
            return None
        parsed.append({"target": target.strip(), "edit_instruction": instruction.strip()})
    return parsed


def parse_refer_objects(text):
    data = load_json(text)
    if isinstance(data, dict):
        data = data.get("refer_object") or data.get("Refer_object")
    if not isinstance(data, list) or len(data) != 5 or not all(isinstance(o, str) and o.strip() for o in data):
        return None
    return [o.strip() for o in data]


def slot_problems(plan, rows):
    problems = []
    for i, (expected, row) in enumerate(zip(plan["repeated_slots"], rows)):
        if row["target"] != expected:
            problems.append(f"slot_instructions[{i}] target mismatch: expected {expected}")
        if expected not in row["edit_instruction"]:
            problems.append(f"slot_instructions[{i}] edit_instruction does not contain target verbatim: {expected}")
    return problems


def refer_problems(instructions, refer_objects):
    problems = []
    if len(set(refer_objects)) != 5:
        problems.append("refer_object contains duplicates")
    for i, (instruction, refer_object) in enumerate(zip(instructions, refer_objects)):
        if refer_object not in instruction:
            problems.append(f"refer_object[{i}] is not exact substring of instruction[{i}]")
    return problems


def feedback(problems):
    return "\n".join(f"- {p}" for p in list(dict.fromkeys(problems))[-6:]) or "- None."


def combine(instructions):
    text = ", and ".join(s.strip().removesuffix(".") for s in instructions)
    return text if text.endswith(".") else text + "."


def ask(vlm, text, max_new_tokens, image=None):
    content = [{"type": "text", "text": text}]
    if image is not None:
        content.insert(0, {"type": "image", "image": image})
    return vlm.chat([[{"role": "user", "content": content}]], max_new_tokens=max_new_tokens)[0]


def main():
    args = parse_args()
    templates = {}
    for key in ("slot", "generator", "extractor"):
        with open(getattr(args, f"{key}_template"), encoding="utf-8") as f:
            templates[key] = f.read().strip()
    with open(args.jsonl, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    vlm = load_vlm(args.vlm, device=args.device, dtype=torch.bfloat16)

    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        for record in tqdm(records):
            name = record["image"]
            image = Image.open(os.path.join(args.image_dir, name)).convert("RGB")

            # Stage 1: slot plan of the repeated instances, from the source prompt only.
            prompt = templates["slot"].replace("{source_prompt}", record["source_prompt"])
            plan = json.loads(ask(vlm, prompt, args.slot_max_new_tokens).strip())
            plan = {
                "repeated_count": plan["repeated_count"],
                "repeated_category": plan["repeated_category"].strip(),
                "repeated_slots": [s.strip() for s in plan["repeated_slots"]],
                "repeated_details": [s.strip() for s in plan["repeated_details"]],
                "preferred_non_repeated_targets": [s.strip() for s in plan["preferred_non_repeated_targets"]],
            }

            # Stage 2: five edit instructions grounded in the image.
            rows, problems = None, []
            for _ in range(args.retries + 1):
                prompt = templates["generator"].replace(
                    "{slot_plan_json}", json.dumps(plan, ensure_ascii=False, indent=2)
                )
                prompt = prompt.replace("{failure_feedback}", feedback(problems)) + f"\n\nIMAGE_FILENAME: {name}\n"
                parsed = parse_slot_instructions(ask(vlm, prompt, args.gen_max_new_tokens, image))
                if parsed is None:
                    problems.append("could not parse 5 slot_instructions")
                    continue
                rows = parsed
                current = slot_problems(plan, rows)
                problems += current
                if not current:
                    break
            if rows is None:
                tqdm.write(f"{name}: no valid edit instructions, skipped")
                continue
            instructions = [row["edit_instruction"] for row in rows]

            # Stage 3: referring expressions copied verbatim from the instructions.
            refer_objects, problems = None, []
            for _ in range(args.retries + 1):
                prompt = templates["extractor"].replace(
                    "{instruction_list_json}", json.dumps(instructions, ensure_ascii=False)
                )
                parsed = parse_refer_objects(
                    ask(vlm, prompt.replace("{failure_feedback}", feedback(problems)), args.ext_max_new_tokens)
                )
                if parsed is None:
                    problems.append("could not parse 5 refer_object")
                    continue
                refer_objects = parsed
                current = refer_problems(instructions, refer_objects)
                problems += current
                if not current:
                    break
            if refer_objects is None:
                tqdm.write(f"{name}: no valid referring expressions, skipped")
                continue

            result = {"image": name, "editing_instruction": combine(instructions), "refer_object": refer_objects}
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
