# Adapted from EditScore (https://github.com/VectorSpaceLab/EditScore), Apache License 2.0.
# Modified for MIRAGE: masked-target prompts, separate SC/PQ scoring and strict output parsing.
import json

import numpy as np

from .prompts import CONTEXT, EDIT_RULE, PQ_RULE, SC_RULE


def parse_output(text):
    """Scores and reasoning from a '{"reasoning": ..., "score": [...]}' response."""
    data = json.loads(text[text.index("{") : text.rindex("}") + 1])
    return [float(s) for s in data["score"]], data["reasoning"]


class EditScore:
    def __init__(self, backbone, model_name_or_path, score_range=25, num_pass=1, seed=42, **backbone_kwargs):
        if backbone == "openai":
            from .mllm_tools.openai import GPT

            self.model = GPT(model_name_or_path, **backbone_kwargs)
        elif backbone == "qwen3vl":
            from .mllm_tools.qwen3vl import Qwen3VL

            self.model = Qwen3VL(model_name_or_path, **backbone_kwargs)
        elif backbone == "qwen3vl_vllm":
            from .mllm_tools.qwen3vl_vllm import Qwen3VL

            self.model = Qwen3VL(model_name_or_path, **backbone_kwargs)
        self.scale = score_range / 10
        self.num_pass = num_pass
        self.seed = seed
        self.sc_prompt = "\n".join([CONTEXT, EDIT_RULE, SC_RULE.replace("10", str(score_range))])
        self.pq_prompt = "\n".join([CONTEXT, PQ_RULE.replace("10", str(score_range))])

    def _run(self, images, prompt):
        inputs = self.model.prepare_input(images, prompt)
        return [parse_output(self.model.inference(inputs, seed=self.seed + i)) for i in range(self.num_pass)]

    def score_sc(self, images, instruction):
        """Prompt following and consistency averaged over passes, plus the last reasoning."""
        passes = self._run(images, self.sc_prompt.replace("<instruction>", instruction))
        prompt_following = np.mean([scores[0] / self.scale for scores, _ in passes])
        consistency = np.mean([scores[1] / self.scale for scores, _ in passes])
        return float(prompt_following), float(consistency), passes[-1][1]

    def score_pq(self, images):
        passes = self._run(images, self.pq_prompt)
        return float(np.mean([min(scores) / self.scale for scores, _ in passes])), passes[-1][1]
