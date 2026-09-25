"""Run MIRAGE on MIRA-Bench downloaded from the Hugging Face Hub.

All other arguments are forwarded to inference.py, e.g.
    python quick_start.py --model flux2_klein9b --output-dir results/flux2_klein9b
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

import inference

if __name__ == "__main__":
    root = Path(snapshot_download("ziqiangoodgood/MIRAGE", repo_type="dataset")) / "benchmark"
    paths = {
        "--image-root": root,
        "--instruction-jsonl": root / "annotations.jsonl",
        "--crop-instruction-jsonl": root / "crops" / "crop_instruction.jsonl",
    }
    argv = sys.argv[1:] + [str(x) for pair in paths.items() for x in pair]
    inference.main(inference.build_parser().parse_args(argv))
