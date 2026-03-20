#!/usr/bin/env python3

import argparse
import json
import os
import random
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


TAG_RE = re.compile(r"^\s*([A-Z0-9_]+)\s*:\s*(.*?)\s*$")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def count_word(n: int) -> str:
    return {3: "three", 4: "four", 5: "five"}[n]


def position_hint(n: int) -> str:
    if n == 3:
        return "left / center / right"
    if n == 4:
        return "leftmost / left-center / right-center / rightmost"
    if n == 5:
        return "leftmost / second from left / center / second from right / rightmost"
    raise ValueError(n)


def build_count_schedule(num_samples: int, rng: random.Random) -> List[int]:
    n3 = num_samples // 2
    n4 = num_samples // 4
    n5 = num_samples - n3 - n4
    arr = [3] * n3 + [4] * n4 + [5] * n5
    rng.shuffle(arr)
    return arr


def format_recent(items: List[str], k: int) -> str:
    tail = items[-k:]
    if not tail:
        return "NONE"
    return "\n".join([f"- {x}" for x in tail])


def format_top(counter: Counter, k: int) -> str:
    if not counter:
        return "NONE"
    return "\n".join([f"- {name} ({cnt})" for name, cnt in counter.most_common(k)])


def format_recent_pairs(pairs: List[Tuple[str, str]], k: int) -> str:
    tail = pairs[-k:]
    if not tail:
        return "NONE"
    return "\n".join([f"- ({c}) @ ({s})" for s, c in tail])


def format_recent_openings(openings: List[str], k: int) -> str:
    tail = openings[-k:]
    if not tail:
        return "NONE"
    return "\n".join([f"- {x}" for x in tail])


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def call_chat_once(
    model_obj,
    tokenizer,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
) -> str:
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model_obj.parameters()).device
    inputs = tokenizer([chat], return_tensors="pt").to(device)

    kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=(temperature > 0),
    )
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)

    with torch.inference_mode():
        output_ids = model_obj.generate(**kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[:, prompt_len:]
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]


def parse_tagged_lines(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ln in text.splitlines():
        m = TAG_RE.match(ln)
        if not m:
            continue
        key = m.group(1).strip().upper()
        val = m.group(2).strip()
        if val:
            out[key] = val
    return out


def strip_bullet_prefix(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[\-\*\u2022]\s*", "", line)
    line = re.sub(r"^\d+[\.\)]\s*", "", line)
    return line.strip()


def parse_pair_batch(text: str) -> List[dict]:
    pairs = []
    for raw in text.splitlines():
        line = strip_bullet_prefix(raw)
        if not line or "||" not in line:
            continue
        left, right = line.split("||", 1)
        cat = left.strip()
        scene = right.strip()
        if cat and scene:
            pairs.append({"object_category": cat, "scene": scene})
    return pairs


def parse_draft_block(text: str) -> Optional[dict]:
    data = parse_tagged_lines(text)
    if "CATEGORY" not in data or "SCENE" not in data or "PROMPT" not in data:
        return None
    return {
        "object_category": data["CATEGORY"],
        "scene": data["SCENE"],
        "source_prompt": data["PROMPT"],
    }


def parse_judge_block(text: str) -> Optional[dict]:
    data = parse_tagged_lines(text)
    if "DECISION" not in data:
        return None

    dec_raw = norm(data["DECISION"])
    if dec_raw.startswith("pass"):
        decision = "PASS"
    elif dec_raw.startswith("fix"):
        decision = "FIX"
    elif dec_raw.startswith("fail"):
        decision = "FAIL"
    else:
        return None

    return {
        "decision": decision,
        "source_prompt": data.get("PROMPT", ""),
        "feedback": data.get("FEEDBACK", ""),
    }


def validate_pair_minimal(pair: dict) -> Optional[str]:
    if not isinstance(pair, dict):
        return "pair_not_dict"
    if not isinstance(pair.get("object_category", None), str) or not norm(pair["object_category"]):
        return "missing_category"
    if not isinstance(pair.get("scene", None), str) or not norm(pair["scene"]):
        return "missing_scene"
    if len(pair["scene"].strip()) < 12:
        return "scene_too_short"
    return None


def validate_candidate_minimal(candidate: dict) -> Optional[str]:
    if not isinstance(candidate, dict):
        return "candidate_not_dict"
    if not isinstance(candidate.get("object_category", None), str) or not norm(candidate["object_category"]):
        return "missing_category"
    if not isinstance(candidate.get("scene", None), str) or not norm(candidate["scene"]):
        return "missing_scene"
    if not isinstance(candidate.get("source_prompt", None), str) or len(candidate["source_prompt"].strip()) < 120:
        return "prompt_too_short"
    return None


def boost_prompt_for_judge(prompt: str) -> str:
    suffix = (
        " The repeated instances and two extra objects are fully visible, very close to the camera, and occupy most of the near foreground. "
        "They are spaced apart with clear gaps so none touch, overlap, or occlude one another."
    )
    p = prompt.strip()
    if p.endswith(suffix.strip()):
        return p
    return p + suffix


def candidate_to_text(candidate: dict, boosted: bool = False) -> str:
    prompt = candidate["source_prompt"]
    if boosted:
        prompt = boost_prompt_for_judge(prompt)
    return "\n".join([
        f"CATEGORY: {candidate['object_category']}",
        f"SCENE: {candidate['scene']}",
        f"PROMPT: {prompt}",
    ])


def pair_to_text(pair: dict) -> str:
    return "\n".join([
        f"CATEGORY: {pair['object_category']}",
        f"SCENE: {pair['scene']}",
    ])


def collect_pair_pool(
    *,
    target_count: int,
    model_obj,
    tokenizer,
    pair_template: str,
    used_pairs: List[Tuple[str, str]],
    used_pair_set,
    used_categories,
    used_scenes,
    cat_counter: Counter,
    scene_counter: Counter,
    count_word_value: str,
    count_num: int,
    top_k: int,
    max_tokens_pair: int,
    pair_batch_size: int,
    pair_temperature: float,
    pair_top_p: float,
    max_repeat_per_category: int,
    forbidden_window: int,
    topk_common_ban: int,
) -> List[dict]:
    collected: List[dict] = []
    pair_fail_streak = 0
    last_error = "NONE"

    while len(collected) < target_count:
        request_n = min(pair_batch_size, target_count - len(collected) + max(4, pair_batch_size // 3))
        temp_pair = min(pair_temperature + 0.03 * pair_fail_streak, 0.96)
        top_p_pair = min(pair_top_p + 0.01 * pair_fail_streak, 0.97)

        prompt = pair_template
        prompt = prompt.replace("{{REQUEST_N}}", str(request_n))
        prompt = prompt.replace("{{COUNT_NUM}}", str(count_num))
        prompt = prompt.replace("{{COUNT_WORD}}", count_word_value)
        prompt = prompt.replace("{{NOVELTY_MODE}}", "strong" if pair_fail_streak == 0 else ("very strong" if pair_fail_streak <= 2 else "extreme"))
        prompt = prompt.replace("{{FORBIDDEN_CATEGORIES}}", format_recent(sorted(used_categories), forbidden_window))
        prompt = prompt.replace("{{FORBIDDEN_SCENES}}", format_recent(sorted(used_scenes), forbidden_window))
        prompt = prompt.replace("{{TOP_CATEGORIES}}", format_top(cat_counter, topk_common_ban))
        prompt = prompt.replace("{{TOP_SCENES}}", format_top(scene_counter, topk_common_ban))
        prompt = prompt.replace("{{RECENT_PAIRS}}", format_recent_pairs(used_pairs, min(60, forbidden_window)))
        prompt = prompt.replace("{{LAST_ERROR}}", last_error)

        raw = call_chat_once(
            model_obj=model_obj,
            tokenizer=tokenizer,
            messages=[
                {"role": "system", "content": "Output only the requested pair lines. No JSON, no markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=temp_pair,
            top_p=top_p_pair,
            top_k=top_k,
            max_tokens=max_tokens_pair,
        )

        candidates = parse_pair_batch(raw)
        accepted_this_round = 0

        for pair in candidates:
            if len(collected) >= target_count:
                break
            err = validate_pair_minimal(pair)
            if err is not None:
                last_error = err
                continue

            cat = norm(pair["object_category"])
            scene = norm(pair["scene"])

            if cat_counter[cat] >= max_repeat_per_category:
                last_error = "category_repeat_cap"
                continue
            if (scene, cat) in used_pair_set:
                last_error = "duplicate_pair"
                continue

            used_pair_set.add((scene, cat))
            used_pairs.append((scene, cat))
            used_categories.add(cat)
            used_scenes.add(scene)
            cat_counter[cat] += 1
            scene_counter[scene] += 1

            collected.append({
                "object_category": pair["object_category"].strip(),
                "scene": pair["scene"].strip(),
            })
            accepted_this_round += 1

        if accepted_this_round == 0:
            pair_fail_streak += 1
        else:
            pair_fail_streak = 0

    return collected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="source_prompts.jsonl")
    ap.add_argument("--out-pairs", type=str, default="pair_pool.jsonl")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--pair-buffer", type=int, default=40)
    ap.add_argument("--seed", type=int, default=31)

    ap.add_argument("--model", type=str, default=os.environ.get("OPENAI_MODEL", "OpenPipe/Qwen3-14B-Instruct"))
    ap.add_argument("--pair-template", type=str, required=True)
    ap.add_argument("--generator-template", type=str, required=True)
    ap.add_argument("--judge-template", type=str, required=True)

    ap.add_argument("--temperature-pair", type=float, default=0.64)
    ap.add_argument("--top-p-pair", type=float, default=0.84)
    ap.add_argument("--temperature-generator", type=float, default=0.74)
    ap.add_argument("--top-p-generator", type=float, default=0.90)
    ap.add_argument("--temperature-judge", type=float, default=0.22)
    ap.add_argument("--top-p-judge", type=float, default=0.80)
    ap.add_argument("--top-k", type=int, default=20)

    ap.add_argument("--max-tokens-pair", type=int, default=600)
    ap.add_argument("--max-tokens-generator", type=int, default=1300)
    ap.add_argument("--max-tokens-judge", type=int, default=1600)

    ap.add_argument("--pair-batch-size", type=int, default=24)
    ap.add_argument("--max-draft-attempts-per-pair", type=int, default=3)
    ap.add_argument("--max-judge-fix-rounds", type=int, default=4)

    ap.add_argument("--forbidden-window", type=int, default=60)
    ap.add_argument("--topk-common-ban", type=int, default=10)
    ap.add_argument("--max-repeat-per-category", type=int, default=4)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    pair_template = load_text(args.pair_template)
    generator_template = load_text(args.generator_template)
    judge_template = load_text(args.judge_template)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_obj = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    count_schedule = build_count_schedule(args.num_samples + args.pair_buffer, rng)
    used_pairs: List[Tuple[str, str]] = []
    used_pair_set = set()
    used_categories = set()
    used_scenes = set()
    cat_counter = Counter()
    scene_counter = Counter()
    used_openings: List[str] = []

    pool_target = args.num_samples + args.pair_buffer
    pair_pool = collect_pair_pool(
        target_count=pool_target,
        model_obj=model_obj,
        tokenizer=tokenizer,
        pair_template=pair_template,
        used_pairs=used_pairs,
        used_pair_set=used_pair_set,
        used_categories=used_categories,
        used_scenes=used_scenes,
        cat_counter=cat_counter,
        scene_counter=scene_counter,
        count_word_value="three-to-five",
        count_num=3,
        top_k=args.top_k,
        max_tokens_pair=args.max_tokens_pair,
        pair_batch_size=args.pair_batch_size,
        pair_temperature=args.temperature_pair,
        pair_top_p=args.top_p_pair,
        max_repeat_per_category=args.max_repeat_per_category,
        forbidden_window=args.forbidden_window,
        topk_common_ban=args.topk_common_ban,
    )

    with open(args.out_pairs, "w", encoding="utf-8") as pf:
        for item in pair_pool:
            pf.write(json.dumps(item, ensure_ascii=False) + "\\n")

    produced = 0
    pair_index = 0

    with open(args.out, "w", encoding="utf-8") as f, tqdm(total=args.num_samples, desc="source_prompt jsonl", unit="sample") as pbar:
        while produced < args.num_samples:
            if pair_index >= len(pair_pool):
                extra_count = max(args.pair_buffer, args.num_samples // 5)
                extra_pairs = collect_pair_pool(
                    target_count=extra_count,
                    model_obj=model_obj,
                    tokenizer=tokenizer,
                    pair_template=pair_template,
                    used_pairs=used_pairs,
                    used_pair_set=used_pair_set,
                    used_categories=used_categories,
                    used_scenes=used_scenes,
                    cat_counter=cat_counter,
                    scene_counter=scene_counter,
                    count_word_value="three-to-five",
                    count_num=3,
                    top_k=args.top_k,
                    max_tokens_pair=args.max_tokens_pair,
                    pair_batch_size=args.pair_batch_size,
                    pair_temperature=args.temperature_pair,
                    pair_top_p=args.top_p_pair,
                    max_repeat_per_category=args.max_repeat_per_category,
                    forbidden_window=args.forbidden_window,
                    topk_common_ban=args.topk_common_ban,
                )
                pair_pool.extend(extra_pairs)
                with open(args.out_pairs, "a", encoding="utf-8") as pf:
                    for item in extra_pairs:
                        pf.write(json.dumps(item, ensure_ascii=False) + "\\n")

            fixed_pair = pair_pool[pair_index]
            target_count = count_schedule[pair_index]
            target_count_word = count_word(target_count)
            pair_index += 1

            accepted = None
            last_error = "NONE"

            for _ in range(args.max_draft_attempts_per_pair):
                gen_prompt = generator_template
                gen_prompt = gen_prompt.replace("{{COUNT_NUM}}", str(target_count))
                gen_prompt = gen_prompt.replace("{{COUNT_WORD}}", target_count_word)
                gen_prompt = gen_prompt.replace("{{POSITION_LABELS}}", position_hint(target_count))
                gen_prompt = gen_prompt.replace("{{NOVELTY_MODE}}", "strong")
                gen_prompt = gen_prompt.replace("{{FIXED_CATEGORY}}", fixed_pair["object_category"])
                gen_prompt = gen_prompt.replace("{{FIXED_SCENE}}", fixed_pair["scene"])
                gen_prompt = gen_prompt.replace("{{RECENT_OPENINGS}}", format_recent_openings(used_openings, 15))
                gen_prompt = gen_prompt.replace("{{LAST_ERROR}}", last_error)

                raw = call_chat_once(
                    model_obj=model_obj,
                    tokenizer=tokenizer,
                    messages=[
                        {"role": "system", "content": "Output only tagged lines. No JSON, no markdown."},
                        {"role": "user", "content": gen_prompt},
                    ],
                    temperature=args.temperature_generator,
                    top_p=args.top_p_generator,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens_generator,
                )

                draft = parse_draft_block(raw)
                if draft is None:
                    last_error = "generator_parse_failed"
                    continue

                current_candidate = {
                    "object_category": fixed_pair["object_category"],
                    "scene": fixed_pair["scene"],
                    "source_prompt": draft["source_prompt"],
                }
                judge_error = "NONE"
                judged_candidate = None

                for _ in range(args.max_judge_fix_rounds):
                    judge_prompt = judge_template
                    judge_prompt = judge_prompt.replace("{{COUNT_NUM}}", str(target_count))
                    judge_prompt = judge_prompt.replace("{{COUNT_WORD}}", target_count_word)
                    judge_prompt = judge_prompt.replace("{{POSITION_LABELS}}", position_hint(target_count))
                    judge_prompt = judge_prompt.replace("{{FIXED_PAIR}}", pair_to_text(fixed_pair))
                    judge_prompt = judge_prompt.replace("{{RECENT_OPENINGS}}", format_recent_openings(used_openings, 15))
                    judge_prompt = judge_prompt.replace("{{CURRENT_CANDIDATE}}", candidate_to_text(current_candidate, boosted=True))
                    judge_prompt = judge_prompt.replace("{{LAST_ERROR}}", judge_error if judge_error else "NONE")

                    judged_raw = call_chat_once(
                        model_obj=model_obj,
                        tokenizer=tokenizer,
                        messages=[
                            {"role": "system", "content": "Output only tagged lines. No JSON, no markdown."},
                            {"role": "user", "content": judge_prompt},
                        ],
                        temperature=args.temperature_judge,
                        top_p=args.top_p_judge,
                        top_k=args.top_k,
                        max_tokens=args.max_tokens_judge,
                    )
                    judged_obj = parse_judge_block(judged_raw)
                    if judged_obj is None:
                        judge_error = "judge_parse_failed"
                        continue

                    if judged_obj["decision"] == "FAIL":
                        judge_error = judged_obj["feedback"] or "judge_fail"
                        judged_candidate = None
                        break

                    candidate = {
                        "object_category": fixed_pair["object_category"],
                        "scene": fixed_pair["scene"],
                        "source_prompt": judged_obj["source_prompt"] or current_candidate["source_prompt"],
                    }
                    cerr = validate_candidate_minimal(candidate)
                    if cerr is not None:
                        judge_error = cerr
                        current_candidate = candidate
                        continue

                    if judged_obj["decision"] == "PASS":
                        judged_candidate = candidate
                        break

                    current_candidate = candidate
                    judge_error = judged_obj["feedback"] or "needs_more_refinement"

                if judged_candidate is None and validate_candidate_minimal(current_candidate) is None:
                    judged_candidate = current_candidate

                if judged_candidate is not None:
                    accepted = judged_candidate
                    break

            if accepted is None:
                print(
                    f"[warn] skipped pair for future image {produced + 1}: "
                    f"({fixed_pair['object_category']}) @ ({fixed_pair['scene']}); last_error={last_error}"
                )
                continue

            image_id = produced + 1
            rec = {"image": f"{image_id}.jpg", "source_prompt": accepted["source_prompt"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
            f.flush()

            opening_stem = " ".join(norm(accepted["source_prompt"]).split()[:4])
            if opening_stem:
                used_openings.append(opening_stem)

            produced += 1
            pbar.update(1)

    print(f"Wrote {args.num_samples} samples to {args.out}")
    print(f"Wrote pair pool to {args.out_pairs}")


if __name__ == "__main__":
    main()
