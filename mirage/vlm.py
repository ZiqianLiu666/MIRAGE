import json
import math
import re

import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    LogitsProcessor,
    LogitsProcessorList,
)

from .geometry import grow


class PresencePenalty(LogitsProcessor):
    """Apply an additive presence penalty to generated tokens, excluding the prompt."""

    def __init__(self, penalty, input_length):
        self.penalty = penalty
        self.input_length = input_length

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.input_length :]
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, generated, True)
        return scores - seen.to(scores.dtype) * self.penalty


def _load_json(text, required_keys=(), allow_array=True, any_keys=()):
    """Read the explicit JSON answer, or a complete container with the expected fields."""
    def matches(data):
        if isinstance(data, dict):
            return all(key in data for key in required_keys) and (not any_keys or any(key in data for key in any_keys))
        if allow_array and isinstance(data, list):
            return all(isinstance(item, dict) and matches(item) for item in data)
        return False

    fenced = re.search(r"```(?:json\b)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        # An explicit answer block takes precedence over coordinates in prose.
        # Decode the entire block so truncated containers cannot be salvaged.
        data = json.loads(fenced[1].strip())
        if matches(data):
            return data
        raise ValueError(f"VLM JSON answer does not match expected fields {required_keys}: {text!r}")
    if "```" in text:
        raise ValueError(f"VLM response contains an incomplete JSON code block: {text!r}")

    decoder = json.JSONDecoder()
    offset = 0
    while match := re.search(r"[\[{]", text[offset:]):
        data, end = decoder.raw_decode(text, offset + match.start())
        if matches(data):
            return data
        # Skip the whole container, never reinterpret a nested object as an answer.
        offset = end
    raise ValueError(f"VLM response contains no JSON object or array with expected fields {required_keys}: {text!r}")


def _grounding_box(values, yx=False):
    """Validate a normalized model box before converting its coordinate order."""
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("A bounding box must contain exactly four coordinates")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Bounding-box coordinates must be finite numbers")
    if any(v < 0 or v > 1000 for v in values):
        raise ValueError("Bounding-box coordinates must be between 0 and 1000")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("A bounding box must have positive width and height")
    a, b, c, d = map(float, values)
    return [b, a, d, c] if yx else [a, b, c, d]


def _parse_grounding_boxes(text, yx=False):
    fenced = re.search(r"```(?:json\b)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    answer = fenced[1].strip() if fenced else re.sub(r"^-\s+", "", text.strip(), count=1)
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        data = None
    # A complete four-number answer is a box, even without a dictionary key.
    # Never extract an unlabelled coordinate vector from explanatory prose.
    if isinstance(data, list) and data and any(not isinstance(item, dict) for item in data):
        return [_grounding_box(data, yx=yx)]

    data = _load_json(text, any_keys=("box_2d", "bbox_2d"))
    items = data if isinstance(data, list) else [data]
    boxes = []
    for item in items:
        keys = [key for key in ("box_2d", "bbox_2d") if key in item]
        if len(keys) != 1:
            raise ValueError("A detection must specify exactly one coordinate convention")
        key = keys[0]
        boxes.append(_grounding_box(item[key], yx=key == "box_2d"))
    return boxes


def to_pixels(box, size, padding=0, min_size=1):
    width, height = size
    x1, y1, x2, y2 = (
        int(round(max(0.0, min(v, 1000.0)) / 1000.0 * total)) for v, total in zip(box, (width, height, width, height))
    )
    x1 = max(0, min(x1 - padding, width - 1))
    y1 = max(0, min(y1 - padding, height - 1))
    x2 = max(0, min(x2 + padding, width))
    y2 = max(0, min(y2 + padding, height))
    x1, x2 = grow(x1, x2, min_size, width)
    y1, y2 = grow(y1, y2, min_size, height)
    return x1, y1, x2, y2


def _locate_prompt(refer_object, box_key, yx):
    coordinates = "[y_min, x_min, y_max, x_max]" if yx else "[x_min, y_min, x_max, y_max]"
    order = "top, left, bottom, right" if yx else "left, top, right, bottom"
    return (
        f"Detect and locate the single visible target described by '{refer_object}'. "
        "For a part or attribute, locate that region only and use the parent to identify the instance. "
        "Spatial relations such as 'to the right of' describe separate objects, not parts. "
        "If several similar instances are present, use all spatial and relational "
        "qualifiers in the phrase to choose exactly one. "
        "The bounding box should cover the complete visible target, without cutting "
        "off visible parts, while excluding unrelated neighboring objects. "
        "Output a JSON array with exactly one object in this format: "
        f'[{{"{box_key}": {coordinates}, "label": "target"}}]. '
        f"Use normalized coordinates from 0 to 1000 in this order: {order}. "
        "Return [] if the target is not visible. "
        "Output JSON only, without Markdown or explanation."
    )


class VLM:
    model_cls = AutoModelForMultimodalLM
    template_kwargs = {"enable_thinking": False}
    generation_tokens = 2048
    grounding_tokens = None
    generation_kwargs = {"do_sample": False, "num_beams": 1}
    text_generation_kwargs = {}
    box_key = "bbox_2d"
    box_yx = False

    def __init__(self, model_id, device, dtype):
        self.device = device
        self.model = self.model_cls.from_pretrained(model_id, dtype=dtype, device_map=device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
        self.generate_kwargs = {"pad_token_id": self.processor.tokenizer.pad_token_id}

    def generation_settings(self, vision=False):
        return {
            **self.generation_kwargs,
            **({} if vision else self.text_generation_kwargs),
        }

    def chat(self, batch, max_new_tokens=None):
        inputs = self.processor.apply_chat_template(
            batch,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            processor_kwargs={"padding": True},
            **self.template_kwargs,
        ).to(self.device)
        vision = "pixel_values" in inputs
        settings = self.generation_settings(vision)
        if max_new_tokens is None:
            max_new_tokens = (self.grounding_tokens if vision else None) or self.generation_tokens
        presence_penalty = settings.pop("presence_penalty", 0.0)
        input_length = inputs["input_ids"].shape[1]
        if presence_penalty:
            settings["logits_processor"] = LogitsProcessorList([PresencePenalty(presence_penalty, input_length)])
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **settings, **self.generate_kwargs)
        generated = output[:, input_length:]
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        self.last_generation_stats = []
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        for input_tokens, token_ids in zip(input_lengths, generated.tolist(), strict=True):
            eos_position = next((i for i, token in enumerate(token_ids) if token in eos_ids), None)
            self.last_generation_stats.append({
                "input_tokens": input_tokens,
                "generated_tokens": len(token_ids) if eos_position is None else eos_position + 1,
                "max_new_tokens": max_new_tokens,
                "ended_with_eos": eos_position is not None,
                "hit_token_limit": eos_position is None and len(token_ids) >= max_new_tokens,
            })
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def grounding_messages(self, image, phrase):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _locate_prompt(phrase, self.box_key, self.box_yx)},
                ],
            },
        ]

    def parse_boxes(self, text):
        return _parse_grounding_boxes(text, yx=self.box_yx)

    def iter_locate(self, images, phrases):
        """Yield validated boxes so callers can save each success before a later error."""
        batch = [self.grounding_messages(image, phrase) for image, phrase in zip(images, phrases, strict=True)]
        if not batch:
            return
        for phrase, text in zip(phrases, self.chat(batch, self.grounding_tokens or self.generation_tokens), strict=True):
            try:
                found = self.parse_boxes(text)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(
                    f"{type(self).__name__} returned invalid grounding for {phrase!r}. "
                    f"Raw response: {text!r}"
                ) from exc
            if not found:
                raise ValueError(
                    f"{type(self).__name__} found no box for {phrase!r}. Raw response: {text!r}"
                )
            if len(found) != 1:
                raise ValueError(f"{type(self).__name__} returned {len(found)} boxes for {phrase!r}. Raw response: {text!r}")
            yield found[0]

    def ground(self, images, phrases):
        return list(self.iter_locate(images, phrases))

    def locate(self, images, phrases):
        return self.ground(images, phrases)


class Qwen35(VLM):
    # Official non-thinking general-task profile, for text and images.
    generation_tokens = 32768
    generation_kwargs = {
        "do_sample": True, "num_beams": 1, "temperature": 0.7,
        "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        "repetition_penalty": 1.0, "presence_penalty": 1.5,
    }


class Gemma4(VLM):
    box_key = "box_2d"
    box_yx = True
    generation_kwargs = {
        "do_sample": True, "num_beams": 1, "temperature": 1.0,
        "top_p": 0.95, "top_k": 64, "repetition_penalty": 1.0,
    }

    def __init__(self, model_id, device, dtype):
        super().__init__(model_id, device, dtype)
        self.processor.image_processor.max_soft_tokens = 1120


BACKENDS = {
    "qwen35": (Qwen35, "Qwen/Qwen3.5-9B"),
    "gemma4": (Gemma4, "google/gemma-4-12B-it"),
}


def load_vlm(name, model_id=None, device="cuda", dtype=torch.bfloat16):
    cls, default_id = BACKENDS[name]
    return cls(model_id or default_id, device, dtype)


def parse_instructions(vlm, instructions):
    system_prompt = """
        You are an information extraction engine for fine-grained grounding in image editing.
        Given an instruction containing one or multiple edits, extract TWO aligned lists: {"Refer_object":[...], "New_edit_instruction":[...]}
        Refer_object:
        - The EXACT visual region to edit.
        - Use the MOST specific referring phrase.
        - Prefer parts/attributes over whole objects.
        - Preserve spatial or relational qualifiers.
        - MUST be a contiguous text span copied from input.
        - MUST refer to something that ALREADY EXISTS in the image.
        - For an ADD edit ("add X to/on/in front of Y"), Refer_object MUST
          be Y (the existing anchor), NEVER X (the object being added,
          since it is not yet in the image and cannot be grounded).
        Do NOT:
        - Generalize or summarize.
        - Collapse to only the object name.

        Counting rule:
        - The instruction is a sequence of atomic edits, usually joined by "and".
        - Output EXACTLY ONE (Refer_object, New_edit_instruction) pair per atomic edit.
        - NEVER split one atomic edit into two pairs, and NEVER merge two
          atomic edits into one pair.

        New_edit_instruction:
        - A complete, self-contained imperative, understandable on its own
          without Refer_object. Never a fragment.
        - Keep the edit action AND the head noun of Refer_object, together with
          the modifiers that describe the object itself: colour, pattern,
          material, size, breed, posture ("long-haired gray cat",
          "black-and-white calico cat", "orange cushion", "seated passenger").
          The region an instruction is applied to often contains more than one
          object of the same kind, and these modifiers are what tells them
          apart. They stay true inside that region, so they are never dropped.
        - Drop only the localizing part of Refer_object: ordinals, positions
          and relations to other objects ("leftmost", "second from the right",
          "in the center", "beneath the middle dog"). Those describe the whole
          image and are false inside the region this instruction is applied to.
        - Rewrite minimally for clarity.
        Rules:
        - Output ONE-LINE valid JSON only.
        - No explanation or markdown.
        - Always choose the most localized editable region.

        Example:
        Input: Change the leftmost bird's feathers to soft down feathers, and add a glass of wine in front of the man sitting on the right side of the image, and remove the first penguin from the left, and replace the kite shaped like a butterfly with a kite shaped like a dragon, and make the texture of the second bird from the right smooth and reflective.
        Output: {"Refer_object":["leftmost bird's feathers", "man sitting on the right side of the image", "first penguin from the left", "kite shaped like a butterfly", "second bird from the right"],
        "New_edit_instruction":["Change the bird's feathers to soft down feathers", "add a glass of wine in front of the man sitting", "remove the penguin", "replace the kite shaped like a butterfly with a kite shaped like a dragon", "make the texture of the bird smooth and reflective"]}

        Now process the user input.
    """
    batch = [
        [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": f"Input: <<<{instruction}>>>\nOutput:"}]},
        ]
        for instruction in instructions
    ]
    parsed = []
    if not batch:
        return parsed
    for instruction, text in zip(instructions, vlm.chat(batch, max_new_tokens=vlm.generation_tokens), strict=True):
        try:
            data = _load_json(text, required_keys=("Refer_object", "New_edit_instruction"), allow_array=False)
            refs, edits = data["Refer_object"], data["New_edit_instruction"]
            # Some models emit a scalar string instead of a one-element list.
            refs = [refs] if isinstance(refs, str) else refs
            edits = [edits] if isinstance(edits, str) else edits
            if any(not isinstance(items, list) or not items for items in (refs, edits)):
                raise ValueError("Instruction outputs must be nonempty lists")
            if any(not isinstance(value, str) or not value.strip() for value in refs + edits):
                raise ValueError("Targets and edit instructions must be nonempty strings")
            parsed.append([(ref.strip(), inst.strip()) for ref, inst in zip(refs, edits, strict=True)])
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(
                f"{type(vlm).__name__} returned invalid edits for {instruction!r}. Raw response: {text!r}"
            ) from exc
    return parsed
