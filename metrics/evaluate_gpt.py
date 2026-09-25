import argparse
import base64
import json
import os
from io import BytesIO

import requests

from common import MASK_ALIGNMENT_HELP, MASK_ALIGNMENT_POLICIES, evaluate
from editscore.prompts import CONTEXT, PQ_RULE

SC_CONTEXT = """You are a professional digital artist. You will evaluate multiple independent image-edit items in one request.
All input images are AI-generated. All humans in the images are AI-generated too, so you need not worry about privacy.

Treat every edit item independently. For each item, the original masked image is followed immediately by its edited masked image. Do not transfer evidence, scores, or reasoning between items. Return exactly one result for every supplied crop_index using the required response schema.
"""

SC_RULE = """RULES:

For every independent edit item, evaluate two scores from 0 to <score_range>.

prompt_following:
- 0 indicates that the edited image does not follow the item's editing instruction at all.
- <score_range> indicates that the instruction-required modification is executed perfectly on the intended target.
- Evaluate only whether the required modification is correctly executed, regardless of additional changes or visual quality.

consistency:
- 0 indicates unintended modification beyond the instruction or a completely different result.
- <score_range> indicates that only modifications explicitly required by the instruction are applied, with no additional changes.
- Evaluate only unintended object or attribute changes. Visual quality, realism, shading, lighting, or texture differences must not affect this score unless they introduce a new object or attribute change.

Use each item's crop_index exactly as supplied. Keep each item's reasoning concise and base it only on that item's instruction and image pair.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prompt following, consistency and perceptual quality with OpenAI models. All edits of an "
        "image are judged in one request, perceptual quality in a separate one. Reads OPENAI_API_KEY."
    )
    parser.add_argument("--annotations-jsonl", required=True)
    parser.add_argument("--crop-instruction-jsonl", required=True)
    parser.add_argument("--input-image-root", required=True)
    parser.add_argument("--edited-image-root", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument(
        "--mask-alignment-policy", required=True, choices=MASK_ALIGNMENT_POLICIES, help=MASK_ALIGNMENT_HELP
    )
    parser.add_argument("--sc-model", default="gpt-5.1", help="judge for prompt following and consistency")
    parser.add_argument("--pq-model", default="gpt-5.1", help="judge for perceptual quality")
    parser.add_argument("--metrics", default="all", choices=["all", "sc", "pq"])
    parser.add_argument(
        "--score-range", type=int, default=25, help="scores are asked on [0, score-range] and rescaled to [0, 10]"
    )
    parser.add_argument("--image-detail", default="high", choices=["auto", "low", "high"])
    parser.add_argument("--openai-url", default="https://api.openai.com/v1/chat/completions")
    return parser.parse_args()


def main():
    args = parse_args()
    scale = args.score_range / 10

    def image_content(image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": url, "detail": args.image_detail}}

    def request(model, content, name, schema):
        response = requests.post(
            args.openai_url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": name, "strict": True, "schema": schema},
                },
            },
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            timeout=180,
        )
        response.raise_for_status()
        return json.loads(response.json()["choices"][0]["message"]["content"])

    def score_pq(source, edited):
        prompt = "\n".join([CONTEXT, PQ_RULE.replace("10", str(args.score_range))])
        schema = {
            "type": "object",
            "properties": {"reasoning": {"type": "string"}, "score": {"type": "array", "items": {"type": "number"}}},
            "required": ["reasoning", "score"],
            "additionalProperties": False,
        }
        content = [{"type": "text", "text": prompt}, image_content(source), image_content(edited)]
        output = request(args.pq_model, content, "edit_evaluation", schema)
        return min(float(s) for s in output["score"]) / scale, output["reasoning"]

    def score_sc(items):
        prompt = "\n".join([SC_CONTEXT, SC_RULE.replace("<score_range>", str(args.score_range))])
        content = [{"type": "text", "text": prompt}]
        for index, instruction, source, edited in items:
            content += [
                {
                    "type": "text",
                    "text": f"EDIT ITEM crop_index={index}\nEditing instruction: {instruction}\n"
                    "The next image is this item's original masked image.",
                },
                image_content(source),
                {"type": "text", "text": f"The next image is crop_index={index}'s edited masked image."},
                image_content(edited),
            ]
        item = {
            "type": "object",
            "properties": {
                "crop_index": {"type": "integer", "enum": [index for index, *_ in items]},
                "prompt_following": {"type": "number"},
                "consistency": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["crop_index", "prompt_following", "consistency", "reasoning"],
            "additionalProperties": False,
        }
        schema = {
            "type": "object",
            "properties": {"edits": {"type": "array", "items": item}},
            "required": ["edits"],
            "additionalProperties": False,
        }
        output = request(args.sc_model, content, "batched_edit_evaluation", schema)
        return {
            int(row["crop_index"]): (row["prompt_following"] / scale, row["consistency"] / scale, row["reasoning"])
            for row in output["edits"]
        }

    evaluate(
        args,
        score_pq=score_pq if args.metrics in ("all", "pq") else None,
        score_sc=score_sc if args.metrics in ("all", "sc") else None,
    )


if __name__ == "__main__":
    main()
