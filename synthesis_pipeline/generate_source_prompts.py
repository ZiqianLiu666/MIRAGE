import argparse
import json
import random
import re
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

COUNT_WORDS = {3: "three", 4: "four", 5: "five"}
POSITION_LABELS = {
    3: "left / center / right",
    4: "leftmost / left-center / right-center / rightmost",
    5: "leftmost / second from left / center / second from right / rightmost",
}
TAGGED_SYSTEM = "Output only tagged lines. No JSON, no markdown."
MIN_PROMPT_CHARS = 120
MIN_SCENE_CHARS = 12
# Appended to the draft shown to the judge so that it checks the layout constraints explicitly.
JUDGE_SUFFIX = (
    " The repeated instances and two extra objects are fully visible, very close to the camera, and occupy most of "
    "the near foreground. They are spaced apart with clear gaps so none touch, overlap, or occlude one another."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate image descriptions with 3-5 repeated instances (Sec. A.1 of the paper)."
    )
    parser.add_argument("--pair-template", required=True)
    parser.add_argument("--generator-template", required=True)
    parser.add_argument("--judge-template", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--pair-buffer", type=int, default=40, help="extra (category, scene) pairs to sample")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--model", default="OpenPipe/Qwen3-14B-Instruct")
    parser.add_argument("--temperature-pair", type=float, default=0.64)
    parser.add_argument("--top-p-pair", type=float, default=0.84)
    parser.add_argument("--temperature-generator", type=float, default=0.74)
    parser.add_argument("--top-p-generator", type=float, default=0.90)
    parser.add_argument("--temperature-judge", type=float, default=0.22)
    parser.add_argument("--top-p-judge", type=float, default=0.80)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens-pair", type=int, default=600)
    parser.add_argument("--max-tokens-generator", type=int, default=1300)
    parser.add_argument("--max-tokens-judge", type=int, default=1600)
    parser.add_argument("--pair-batch-size", type=int, default=24)
    parser.add_argument("--max-draft-attempts", type=int, default=3)
    parser.add_argument("--max-judge-rounds", type=int, default=4)
    parser.add_argument("--forbidden-window", type=int, default=60)
    parser.add_argument("--topk-common-ban", type=int, default=10)
    parser.add_argument("--max-repeat-per-category", type=int, default=4)
    return parser.parse_args()


def norm(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def fill(template, **values):
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def bullets(items):
    return "\n".join(f"- {item}" for item in items) if items else "NONE"


def parse_tags(text):
    tags = {}
    for line in text.splitlines():
        match = re.match(r"^\s*([A-Z0-9_]+)\s*:\s*(.*?)\s*$", line)
        if match and match[2]:
            tags[match[1].upper()] = match[2]
    return tags


def parse_pairs(text):
    pairs = []
    for line in text.splitlines():
        line = re.sub(r"^\d+[\.\)]\s*", "", re.sub(r"^[\-\*•]\s*", "", line.strip())).strip()
        if "||" in line:
            category, scene = (part.strip() for part in line.split("||", 1))
            if category and scene:
                pairs.append((category, scene))
    return pairs


class Generator:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto").eval()
        self.used_pairs = []
        self.categories = Counter()
        self.scenes = Counter()
        self.openings = []

    @torch.no_grad()
    def chat(self, system, prompt, temperature, top_p, max_tokens):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=self.args.top_k,
        )
        return self.tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]

    def sample_pairs(self, template, count):
        """Sample new (category, scene) pairs, raising the temperature while batches keep failing."""
        args = self.args
        pairs = []
        fail_streak = 0
        last_error = "NONE"
        with tqdm(total=count, desc="pairs") as progress:
            while len(pairs) < count:
                prompt = fill(
                    template,
                    REQUEST_N=min(args.pair_batch_size, count - len(pairs) + max(4, args.pair_batch_size // 3)),
                    NOVELTY_MODE="strong" if fail_streak == 0 else "very strong" if fail_streak <= 2 else "extreme",
                    FORBIDDEN_CATEGORIES=bullets(sorted(self.categories)[-args.forbidden_window :]),
                    FORBIDDEN_SCENES=bullets(sorted(self.scenes)[-args.forbidden_window :]),
                    TOP_CATEGORIES=bullets(
                        [f"{c} ({n})" for c, n in self.categories.most_common(args.topk_common_ban)]
                    ),
                    TOP_SCENES=bullets([f"{s} ({n})" for s, n in self.scenes.most_common(args.topk_common_ban)]),
                    RECENT_PAIRS=bullets(
                        [f"({c}) @ ({s})" for s, c in self.used_pairs[-min(60, args.forbidden_window) :]]
                    ),
                    LAST_ERROR=last_error,
                )
                text = self.chat(
                    "Output only the requested pair lines. No JSON, no markdown.",
                    prompt,
                    min(args.temperature_pair + 0.03 * fail_streak, 0.96),
                    min(args.top_p_pair + 0.01 * fail_streak, 0.97),
                    args.max_tokens_pair,
                )
                accepted = 0
                for category, scene in parse_pairs(text):
                    if len(pairs) >= count:
                        break
                    key = (norm(scene), norm(category))
                    if len(scene) < MIN_SCENE_CHARS:
                        last_error = "scene_too_short"
                    elif self.categories[key[1]] >= args.max_repeat_per_category:
                        last_error = "category_repeat_cap"
                    elif key in self.used_pairs:
                        last_error = "duplicate_pair"
                    else:
                        self.used_pairs.append(key)
                        self.categories[key[1]] += 1
                        self.scenes[key[0]] += 1
                        pairs.append((category, scene))
                        accepted += 1
                fail_streak = 0 if accepted else fail_streak + 1
                progress.update(accepted)
                progress.set_postfix_str(f"fail_streak={fail_streak}, last_error={last_error}")
        return pairs

    def describe(self, generator_template, judge_template, category, scene, count):
        """Draft a description for a fixed pair and refine it with the judge; None if it never passes."""
        args = self.args
        shared = dict(
            COUNT_NUM=count,
            COUNT_WORD=COUNT_WORDS[count],
            POSITION_LABELS=POSITION_LABELS[count],
            RECENT_OPENINGS=bullets(self.openings[-15:]),
        )
        last_error = "NONE"
        for _ in range(args.max_draft_attempts):
            prompt = fill(
                generator_template, FIXED_CATEGORY=category, FIXED_SCENE=scene, LAST_ERROR=last_error, **shared
            )
            draft = parse_tags(
                self.chat(
                    TAGGED_SYSTEM, prompt, args.temperature_generator, args.top_p_generator, args.max_tokens_generator
                )
            )
            if not {"CATEGORY", "SCENE", "PROMPT"} <= draft.keys():
                last_error = "generator_parse_failed"
                continue

            candidate = draft["PROMPT"]
            judge_error = "NONE"
            for _ in range(args.max_judge_rounds):
                shown = candidate.strip()
                if not shown.endswith(JUDGE_SUFFIX.strip()):
                    shown += JUDGE_SUFFIX
                prompt = fill(
                    judge_template,
                    FIXED_PAIR=f"CATEGORY: {category}\nSCENE: {scene}",
                    CURRENT_CANDIDATE=f"CATEGORY: {category}\nSCENE: {scene}\nPROMPT: {shown}",
                    LAST_ERROR=judge_error,
                    **shared,
                )
                verdict = parse_tags(
                    self.chat(TAGGED_SYSTEM, prompt, args.temperature_judge, args.top_p_judge, args.max_tokens_judge)
                )
                decision = norm(verdict.get("DECISION", ""))
                if not decision.startswith(("pass", "fix", "fail")):
                    judge_error = "judge_parse_failed"
                    continue
                if decision.startswith("fail"):
                    judge_error = verdict.get("FEEDBACK") or "judge_fail"
                    break
                candidate = verdict.get("PROMPT") or candidate
                if len(candidate.strip()) < MIN_PROMPT_CHARS:
                    judge_error = "prompt_too_short"
                elif decision.startswith("pass"):
                    return candidate
                else:
                    judge_error = verdict.get("FEEDBACK") or "needs_more_refinement"
            last_error = judge_error
        return None


def main():
    args = parse_args()
    with open(args.pair_template, encoding="utf-8") as f:
        pair_template = f.read()
    with open(args.generator_template, encoding="utf-8") as f:
        generator_template = f.read()
    with open(args.judge_template, encoding="utf-8") as f:
        judge_template = f.read()

    # 50% of the images get three instances, 25% four and 25% five.
    counts = [3] * (args.num_samples // 2) + [4] * (args.num_samples // 4)
    counts += [5] * (args.num_samples - len(counts))
    random.Random(args.seed).shuffle(counts)

    generator = Generator(args)
    pairs = generator.sample_pairs(pair_template, args.num_samples + args.pair_buffer)
    produced = 0
    with open(args.out, "w", encoding="utf-8") as f, tqdm(total=args.num_samples, desc="descriptions") as progress:
        while produced < args.num_samples:
            if not pairs:
                pairs = generator.sample_pairs(pair_template, max(args.pair_buffer, args.num_samples // 5))
            category, scene = pairs.pop(0)
            description = generator.describe(generator_template, judge_template, category, scene, counts[produced])
            if description is None:
                tqdm.write(f"discarded ({category}) @ ({scene})")
                continue
            produced += 1
            f.write(json.dumps({"image": f"{produced}.jpg", "source_prompt": description}, ensure_ascii=False) + "\n")
            f.flush()
            generator.openings.append(" ".join(norm(description).split()[:4]))
            progress.update(1)


if __name__ == "__main__":
    main()
