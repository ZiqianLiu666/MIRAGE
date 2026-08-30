import random

from .json_parser import parse_vlm_output_to_dict


def mllm_output_to_dict(
    input_string,
    give_up_parsing=False,
    text_prompt=None,
    score_range: int = 10,
):
    if input_string == "rate_limit_exceeded":
        return input_string
    if give_up_parsing:
        score = random.randint(0, score_range)
        return {
            "score": [score, score],
            "reasoning": f"guess_if_cannot_parse | {input_string}",
        }

    result = parse_vlm_output_to_dict(input_string)
    return result if result["score"] else False
