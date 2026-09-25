import json
import re

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from .geometry import grow

# Two-stage localization for "X of Y" / "Y's X" phrases: find the parent Y first,
# then look for the part X inside a crop around it.
PART_PADDING = 32
PART_MIN_SIZE = 192
PART_MAX_WORDS = 8


def _load_json(text):
    """First JSON value in a model response, e.g. inside a markdown code block."""
    return json.JSONDecoder().raw_decode(text, re.search(r"[\[{]", text).start())[0]


def _json_items(text):
    data = _load_json(text)
    return data if isinstance(data, list) else [data]


def _box(values, yx=False):
    a, b, c, d = (float(v) for v in values)
    x1, y1, x2, y2 = (b, a, d, c) if yx else (a, b, c, d)
    return [
        max(0.0, min(1000.0, min(x1, x2))),
        max(0.0, min(1000.0, min(y1, y2))),
        max(0.0, min(1000.0, max(x1, x2))),
        max(0.0, min(1000.0, max(y1, y2))),
    ]


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


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _needs_part_search(direct, parent):
    """Whether the direct box is missing, covers most of the parent or lies outside it."""
    if direct is None or _area(direct) <= 0 or _area(parent) <= 0:
        return True
    if _area(direct) / _area(parent) >= 0.6:
        return True
    overlap_w = max(0.0, min(direct[2], parent[2]) - max(direct[0], parent[0]))
    overlap_h = max(0.0, min(direct[3], parent[3]) - max(direct[1], parent[1]))
    return overlap_w * overlap_h / _area(direct) < 0.5


def _crop_to_image(box, crop_box, size):
    x1, y1, x2, y2 = crop_box
    return [
        (x1 + box[0] / 1000.0 * (x2 - x1)) / size[0] * 1000.0,
        (y1 + box[1] / 1000.0 * (y2 - y1)) / size[1] * 1000.0,
        (x1 + box[2] / 1000.0 * (x2 - x1)) / size[0] * 1000.0,
        (y1 + box[3] / 1000.0 * (y2 - y1)) / size[1] * 1000.0,
    ]


def _clean_phrase(text):
    return re.sub(r"\s+", " ", text.strip().strip(".")).strip()


def _strip_article(text):
    return re.sub(r"^(?:the|a|an)\s+", "", _clean_phrase(text), flags=re.IGNORECASE)


def _split_part(phrase):
    """(parent, part) for phrases like "the dog's ears" or "the ears of the dog"."""
    phrase = _clean_phrase(phrase)
    patterns = [
        re.match(r"^(?P<parent>.+?)'s\s+(?P<part>.+)$", phrase),
        re.match(r"^(?:the|a|an)?\s*(?P<part>.+?)\s+of\s+(?P<parent>.+)$", phrase, flags=re.IGNORECASE),
    ]
    for match in patterns:
        if match:
            parent, part = _strip_article(match["parent"]), _strip_article(match["part"])
            if parent and part and len(part.split()) <= PART_MAX_WORDS:
                return parent, part
    return None


def _locate_prompt(refer_object, mode, box_format):
    if mode == "parent":
        target_rules = """
        What to localize:
        - Box the full visible parent instance described by the phrase.
        - Do NOT box only a small part of the parent.
        - Preserve instance qualifiers such as leftmost, rightmost, middle, lower-left, etc.
        """.strip()
    elif mode == "part":
        target_rules = """
        What to localize:
        - The image is a local crop around the parent object.
        - Box ONLY the requested part/attribute region inside this crop.
        - If the target is a pair or group such as eyes, shoes, headlights, or wheels, box the visible pair/group together.
        - Do NOT box the whole parent object unless the requested part covers most of it.
        - Coordinates must be relative to this cropped image, not the original full image.
        """.strip()
    else:
        target_rules = """
        What to localize:
        - If the phrase has the form "X of Y" or "Y's X",
        the target to box is X (the part), and Y is only used to find which instance.
        Example: "glasses of the person on the leftmost" -> box the glasses only, NOT the whole person.
        Example: "shoes of the boy on the left" -> box the shoes only.
        - Otherwise, box the main noun described by the phrase.
        """.strip()

    return f"""
        You are a precise object detector for referring expressions.

        Task:
        - You are given ONE image and ONE short phrase that describes ONE target.
        - Find the SINGLE visible target region that the phrase refers to
        and output its bounding box.

        {target_rules}

        Disambiguation:
        - If multiple similar objects exist, use words like
        "leftmost", "rightmost", "middle", "center",
        "between A and B", "on the left/right", etc. to choose exactly one.

        Tightness requirements:
        - The box must enclose only the visible target and its immediately necessary visual context.
        - Do NOT include unrelated neighboring instances.
        - Keep minimal padding around the object.

        {box_format}

        STRICT JSON RULES:
        - Use double quotes "..." around all keys and string values.
        - Use colon ":" between keys and values.
        - No trailing commas.

        Target phrase: {refer_object}
        """.strip()


def _qwen35_locate_prompt(refer_object, mode):
    if mode == "parent":
        target = (
            f"the full visible parent instance described by '{refer_object}'. "
            "Preserve spatial qualifiers and box the whole parent, not a small part"
        )
    elif mode == "part":
        target = (
            f"the visible part '{refer_object}' inside this cropped parent image. "
            "Box only that part; if it is a visible pair/group, box the pair/group together"
        )
    else:
        target = (
            f"the single visible target described by '{refer_object}'. "
            "If the phrase is 'X of Y' or Y's X, locate X only and use Y only "
            "to identify the correct instance"
        )
    return (
        f"Detect and locate {target}. "
        "If several similar instances are present, use all spatial and relational "
        "qualifiers in the phrase to choose exactly one. "
        "The bounding box should cover the complete visible target, without cutting "
        "off visible parts, while excluding unrelated neighboring objects. "
        "Output a JSON array with exactly one object in this format: "
        '[{"label": "target", "bbox_2d": [x1, y1, x2, y2]}]. '
        "bbox_2d uses normalized coordinates from 0 to 1000, with (x1,y1) the "
        "top-left and (x2,y2) the bottom-right. Output JSON only."
    )


class VLM:
    model_cls = AutoModelForMultimodalLM
    template_kwargs = {"enable_thinking": False}
    grounding_tokens = 128

    def __init__(self, model_id, device, dtype):
        self.device = device
        self.model = self.model_cls.from_pretrained(model_id, dtype=dtype, device_map=device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
        self.generate_kwargs = {}

    def chat(self, batch, max_new_tokens=512):
        inputs = self.processor.apply_chat_template(
            batch,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            processor_kwargs={"padding": True},
            **self.template_kwargs,
        ).to(self.device)
        output = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, **self.generate_kwargs
        )
        return self.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def grounding_messages(self, image, phrase, mode):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a visual grounding engine. Return only valid JSON."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _locate_prompt(phrase, mode, self.box_format)},
                ],
            },
        ]

    def ground(self, images, phrases, mode="direct"):
        batch = [self.grounding_messages(image, phrase, mode) for image, phrase in zip(images, phrases)]
        boxes = [self.parse_boxes(text) for text in self.chat(batch, self.grounding_tokens)]
        return [found[0] if found else None for found in boxes]

    def locate(self, images, phrases):
        """Normalized [x1, y1, x2, y2] box of each referring expression, or None if not found."""
        boxes = self.ground(images, phrases)
        parts = []
        for i, phrase in enumerate(phrases):
            split = _split_part(phrase)
            if split:
                parts.append((i, *split))
        if not parts:
            return boxes

        parents = self.ground([images[i] for i, _, _ in parts], [parent for _, parent, _ in parts], "parent")
        searches = []
        for (i, _, part), parent in zip(parts, parents):
            if parent is not None and _needs_part_search(boxes[i], parent):
                crop_box = to_pixels(parent, images[i].size, PART_PADDING, PART_MIN_SIZE)
                searches.append((i, part, parent, crop_box))
        if not searches:
            return boxes

        found = self.ground(
            [images[i].crop(crop_box) for i, _, _, crop_box in searches],
            [part for _, part, _, _ in searches],
            "part",
        )
        for (i, _, parent, crop_box), box in zip(searches, found):
            if box is not None:
                boxes[i] = _crop_to_image(box, crop_box, images[i].size)
            elif boxes[i] is None:
                boxes[i] = parent
        return boxes


class Qwen3VL(VLM):
    model_cls = Qwen3VLForConditionalGeneration
    template_kwargs = {}
    box_format = """
        Coordinate system and output:
        - Use NORMALIZED coordinates in the range [0, 1000].
        - 0 = left/top image edge, 1000 = right/bottom image edge.
        - Output integers only. Do NOT output pixel coordinates or percentages.
        - Return ONLY ONE valid JSON array and NOTHING ELSE.
        - JSON format (must match EXACTLY):
          [{"bbox_2d": [x_min, y_min, x_max, y_max], "label": "object"}]
        """.strip()

    def __init__(self, model_id, device, dtype):
        super().__init__(model_id, device, dtype)
        pixels_per_token = 32 * 32
        self.processor.image_processor.size = {
            "shortest_edge": 768 * pixels_per_token,
            "longest_edge": 3072 * pixels_per_token,
        }

    def parse_boxes(self, text):
        return [_box(item["bbox_2d"]) for item in _json_items(text)]


class Qwen35(VLM):
    def __init__(self, model_id, device, dtype):
        super().__init__(model_id, device, dtype)
        self.generate_kwargs = {"pad_token_id": self.processor.tokenizer.pad_token_id}

    def grounding_messages(self, image, phrase, mode):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _qwen35_locate_prompt(phrase, mode)},
                ],
            }
        ]

    def parse_boxes(self, text):
        return [_box(item["bbox_2d"]) for item in _json_items(text)]


class Gemma4(VLM):
    grounding_tokens = 256
    box_format = """
        Coordinate system and output:
        - Use Gemma's native normalized 1000x1000 object-detection coordinates.
        - Return coordinates in this exact order: [y_min, x_min, y_max, x_max].
        - Output integers only. Do NOT output pixel coordinates or percentages.
        - Return ONLY ONE valid JSON array and NOTHING ELSE.
        - JSON format (must match EXACTLY):
          [{"box_2d": [y_min, x_min, y_max, x_max], "label": "object"}]
        """.strip()

    def __init__(self, model_id, device, dtype):
        super().__init__(model_id, device, dtype)
        self.processor.image_processor.max_soft_tokens = 1120

    def parse_boxes(self, text):
        return [_box(item["box_2d"], yx=True) for item in _json_items(text)]


class RegionReasoner(VLM):
    model_cls = Qwen2_5_VLForConditionalGeneration
    image_size = 840
    template = (
        'Please find "{Question}" with bboxs and points.\n'
        "If a reference bbox is provided (e.g., "
        "'above/ below/ to the left of/ to the right of/ inside/ overlapping with/ touching bbox=[x1,y1,x2,y2]'), "
        "use it as spatial guidance only.\n\n"
        "1) In <scene> </scene>, give a concise global scene description.\n"
        "2) If a reference bbox exists, in <focus> </focus> describe ONLY what is visible inside that bbox "
        "(do not output the final answer or target label here).\n"
        "3) In <think> </think>, reason over the whole image by combining the global scene and the reference bbox relation. "
        "Explicitly state which spatial relation from the question you apply (e.g., 'target is above the reference'), "
        "and use it to constrain the search over the scene to locate the target object(s). "
        "If multiple candidates exist, compare them and pick the closest match.\n"
        "4) In <answer> </answer>, output the bbox(es) and point(s) for the target object(s) in JSON.\n\n"
        "Format:\n"
        "<scene> global scene description </scene>\n"
        "<focus> description of reference bbox content (if provided bbox=[x1,y1,x2,y2]) </focus>\n"
        "<think> reasoning that applies the spatial relation to the scene and narrows to the final target(s) </think>\n"
        "<answer>{Answer}</answer>"
    )

    def _generate(self, batch, images=None, **generate_kwargs):
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, do_sample=False, **generate_kwargs)
        return self.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def chat(self, batch, max_new_tokens=512):
        return self._generate(
            batch, max_new_tokens=max_new_tokens, num_beams=1, pad_token_id=self.processor.tokenizer.pad_token_id
        )

    def parse_boxes(self, text):
        answer = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)[1]
        scale = 1000.0 / self.image_size
        return [_box([float(v) * scale for v in item["bbox_2d"]]) for item in _json_items(answer)]

    def locate(self, images, phrases):
        batch = []
        for image, phrase in zip(images, phrases):
            prompt = self.template.format(
                Question=phrase.lower().strip('."?!'),
                Answer='[{"bbox_2d": [10,100,200,210], "point_2d": [30,110]}]',
            )
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
            batch.append([{"role": "user", "content": content}])
        outputs = self._generate(batch, process_vision_info(batch)[0], use_cache=True, max_new_tokens=2048)
        return [found[0] if found else None for found in map(self.parse_boxes, outputs)]


BACKENDS = {
    "qwen35": (Qwen35, "Qwen/Qwen3.5-9B"),
    "gemma4": (Gemma4, "google/gemma-4-12B-it"),
    "qwen8b": (Qwen3VL, "Qwen/Qwen3-VL-8B-Instruct"),
    "qwen4b": (Qwen3VL, "Qwen/Qwen3-VL-4B-Instruct"),
    "regionreasoner": (RegionReasoner, "lmsdss/RegionReasoner-7B"),
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
        "New_edit_instruction":["Change the bird's feathers to soft down feathers", "add a glass of wine in front of the man", "remove the penguin", "replace the kite with a kite shaped like a dragon", "make the texture of the bird smooth and reflective"]}

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
    for text in vlm.chat(batch, max_new_tokens=512):
        data = _load_json(text)
        pairs = zip(data["Refer_object"], data["New_edit_instruction"], strict=True)
        parsed.append([(ref.strip(), inst.strip()) for ref, inst in pairs])
    return parsed
