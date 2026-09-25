import argparse
import os

from common import MASK_ALIGNMENT_HELP, MASK_ALIGNMENT_POLICIES, evaluate
from editscore import EditScore


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prompt following and consistency with an EditScore model (averaged over --num-pass "
        "samples) and perceptual quality with an OpenAI model. Reads OPENAI_API_KEY."
    )
    parser.add_argument("--annotations-jsonl", required=True)
    parser.add_argument("--crop-instruction-jsonl", required=True)
    parser.add_argument("--input-image-root", required=True)
    parser.add_argument("--edited-image-root", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument(
        "--mask-alignment-policy", required=True, choices=MASK_ALIGNMENT_POLICIES, help=MASK_ALIGNMENT_HELP
    )
    parser.add_argument("--sc-backbone", default="qwen3vl_vllm", choices=["qwen3vl", "qwen3vl_vllm"])
    parser.add_argument("--sc-model-name-or-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--sc-lora-path", help="EditScore LoRA, e.g. EditScore/EditScore-Qwen3-VL-8B-Instruct")
    parser.add_argument("--pq-model", default="gpt-5.1")
    parser.add_argument("--openai-url", default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--num-pass", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--score-range", type=int, default=25)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--cache-dir", help="where the merged LoRA model is cached for vLLM")
    return parser.parse_args()


def main():
    args = parse_args()
    vllm_kwargs = {}
    if args.sc_backbone == "qwen3vl_vllm":
        vllm_kwargs = dict(
            cache_dir=args.cache_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
        )
    sc_scorer = EditScore(
        args.sc_backbone,
        args.sc_model_name_or_path,
        score_range=args.score_range,
        num_pass=args.num_pass,
        temperature=args.temperature,
        lora_path=args.sc_lora_path,
        **vllm_kwargs,
    )
    pq_scorer = EditScore(
        "openai",
        args.pq_model,
        score_range=args.score_range,
        api_key=os.environ["OPENAI_API_KEY"],
        url=args.openai_url,
    )

    def score_sc(items):
        return {index: sc_scorer.score_sc([source, edited], text) for index, text, source, edited in items}

    evaluate(args, score_pq=lambda source, edited: pq_scorer.score_pq([source, edited]), score_sc=score_sc)


if __name__ == "__main__":
    main()
