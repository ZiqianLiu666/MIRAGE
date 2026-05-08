import os
import json
import re
from typing import List, Optional
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import torch
from PIL import Image

VLM_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
VLM_MIN_VISUAL_TOKENS = 768
VLM_MAX_VISUAL_TOKENS = 3072
QWEN3_VL_SPATIAL_COMPRESSION = 32
PART_REFINEMENT_PADDING = 32
PART_REFINEMENT_MIN_SIZE = 192

PART_REFINEMENT_TERMS = {
    "arm",
    "arms",
    "back",
    "beard",
    "beak",
    "belly",
    "belt",
    "body",
    "chest",
    "collar",
    "display",
    "dress",
    "ear",
    "ears",
    "eye",
    "eyes",
    "face",
    "feather",
    "feathers",
    "foot",
    "feet",
    "front",
    "fur",
    "hair",
    "halter",
    "hand",
    "hands",
    "handle",
    "handles",
    "hat",
    "head",
    "headlight",
    "headlights",
    "helmet",
    "horn",
    "horns",
    "jacket",
    "lanyard",
    "leg",
    "legs",
    "logo",
    "mouth",
    "neck",
    "nose",
    "pants",
    "paw",
    "paws",
    "patch",
    "scarf",
    "screen",
    "shirt",
    "shell",
    "shoe",
    "shoes",
    "shoulder",
    "shoulders",
    "skin",
    "sleeve",
    "sleeves",
    "tail",
    "tire",
    "tires",
    "wheel",
    "wheels",
    "waist",
    "wing",
    "wings",
    "window",
    "windows",
    "wristband",
}

_backend = None

class _QwenBackend:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID,
            torch_dtype=self.dtype,
        )
        self.model = self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
        self.processor.tokenizer.padding_side = "left"
        self.processor.image_processor.size = {
            "shortest_edge": VLM_MIN_VISUAL_TOKENS * QWEN3_VL_SPATIAL_COMPRESSION * QWEN3_VL_SPATIAL_COMPRESSION,
            "longest_edge": VLM_MAX_VISUAL_TOKENS * QWEN3_VL_SPATIAL_COMPRESSION * QWEN3_VL_SPATIAL_COMPRESSION,
        }

    def chat_batch(self, batch_messages, max_new_tokens=512):
        inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

def get_backend():
    global _backend
    if _backend is None:
        _backend = _QwenBackend()
    return _backend


def _parse_items_from_text(text: str):
    data = json.loads(text.strip())
    return [
        {
            "Refer_object": str(ref).strip(),
            "New_edit_instruction": str(inst).strip(),
        }
        for ref, inst in zip(data["Refer_object"], data["New_edit_instruction"])
    ]


def parse_edit_instruction_batch(edit_instructions):
    system_prompt = """
        You are an information extraction engine for fine-grained grounding in image editing.
        Given an instruction containing one or multiple edits, extract TWO aligned lists: {"Refer_object":[...], "New_edit_instruction":[...]}
        Refer_object:
        - The EXACT visual region to edit.
        - Use the MOST specific referring phrase.
        - Prefer parts/attributes over whole objects.
        - Preserve spatial or relational qualifiers.
        - MUST be a contiguous text span copied from input.
        Do NOT:
        - Generalize or summarize.
        - Collapse to only the object name.
        
        New_edit_instruction:
        - The edit applied ONLY inside Refer_object.
        - Keep the edit action.
        - Remove spatial/localization phrases.
        - Do NOT repeat the full refer phrase.
        - Rewrite minimally for clarity.
        Rules:
        - Output ONE-LINE valid JSON only.
        - No explanation or markdown.
        - Always choose the most localized editable region.

        Example:
        Input: Change the leftmost bird's feathers to soft down feathers, and add a glass of wine in front of the man sitting on the right side of the image, and remove the first penguin from the left, and replace the kite shaped like a butterfly with a kite shaped like a dragon, and make the texture of the second bird from the right smooth and reflective.
        Output: {"Refer_object":["leftmost bird's feathers", "man sitting on the right side of the image", "first penguin from the left", "kite shaped like a butterfly", "second bird from the right"],
        "New_edit_instruction":["Change the bird's feathers to soft down feathers", "add a glass of wine in front of the man", "remove the penguin", "replace the kite shaped like a dragon", "make the texture of bird smooth and reflective"]}

        Now process the user input.
    """

    batch_messages = []
    for instruction in edit_instructions:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Input: <<<{instruction}>>>\nOutput:",
                    }
                ],
            },
        ]
        batch_messages.append(messages)

    outputs = get_backend().chat_batch(batch_messages, max_new_tokens=512)
    return [_parse_items_from_text(text) for text in outputs]


def parse_edit_instruction(edit_instruction: str):
    return parse_edit_instruction_batch([edit_instruction])[0]


def _parse_bboxes_from_text(text: str):
    return [
        {"bbox_2d": [float(v) for v in item["bbox_2d"]]}
        for item in json.loads(text.strip())
    ]

def _clean_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().strip(".")).strip()


def _strip_leading_article(text: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", _clean_phrase(text), flags=re.IGNORECASE)


def _has_refinable_part_term(part_phrase: str) -> bool:
    words = set(re.findall(r"[a-z]+", part_phrase.lower()))
    return bool(words & PART_REFINEMENT_TERMS)


def _extract_part_relation(refer_object: str):
    phrase = _clean_phrase(refer_object)

    possessive = re.match(r"^(?P<parent>.+?)'s\s+(?P<part>.+)$", phrase)
    if possessive:
        parent = _strip_leading_article(possessive.group("parent"))
        part = _strip_leading_article(possessive.group("part"))
        if parent and part and _has_refinable_part_term(part):
            return parent, part

    of_relation = re.match(
        r"^(?:the|a|an)?\s*(?P<part>.+?)\s+of\s+(?P<parent>.+)$",
        phrase,
        flags=re.IGNORECASE,
    )
    if of_relation:
        parent = _strip_leading_article(of_relation.group("parent"))
        part = _strip_leading_article(of_relation.group("part"))
        if parent and part and _has_refinable_part_term(part):
            return parent, part

    return None


def _build_locate_prompt(refer_object: str, prompt_mode: str = "direct") -> str:
    if prompt_mode == "parent":
        target_rules = """
        What to localize:
        - Box the full visible parent instance described by the phrase.
        - Do NOT box only a small part of the parent.
        - Preserve instance qualifiers such as leftmost, rightmost, middle, lower-left, etc.
        """.strip()
    elif prompt_mode == "part":
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

        Coordinate system:
        - Use NORMALIZED coordinates in the range [0, 1000].
        - 0 = left/top image edge, 1000 = right/bottom image edge.
        - Output integers only.
        - Do NOT output pixel coordinates or percentages.

        Output:
        - Return ONLY ONE valid JSON object and NOTHING ELSE.
        - JSON format (must match EXACTLY):
        [
        {{"bbox_2d": [x_min, y_min, x_max, y_max], "label": "object"}}
        ]

        STRICT JSON RULES:
        - Use double quotes "..." around all keys and string values.
        - Use colon ":" between keys and values.
        - No trailing commas.

        Target phrase: {refer_object}
        """.strip()


def _locate_refer_object_batch_direct(image_inputs, refer_objects, prompt_mode: str = "direct"):
    batch_messages = []
    for image_input, refer_object in zip(image_inputs, refer_objects):
        user_prompt = _build_locate_prompt(refer_object, prompt_mode=prompt_mode)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a visual grounding engine. Return only valid JSON.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_input},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        batch_messages.append(messages)

    outputs = get_backend().chat_batch(batch_messages, max_new_tokens=128)
    return [_parse_bboxes_from_text(text) for text in outputs]


def _expand_to_min_size(a1: int, a2: int, limit: int, target: int):
    size = a2 - a1
    if size >= target:
        return a1, a2
    extra = target - size
    left = extra // 2
    right = extra - left
    a1 -= left
    a2 += right
    if a1 < 0:
        a2 += -a1
        a1 = 0
    if a2 > limit:
        a1 -= a2 - limit
        a2 = limit
        if a1 < 0:
            a1 = 0
    return a1, a2


def _bbox_norm_to_pixels(image: Image.Image, bbox, padding: int = 0, min_size: int = 1):
    x1_n, y1_n, x2_n, y2_n = [float(v) for v in bbox["bbox_2d"]]
    width, height = image.size

    def norm1000_to_pixel(a, total):
        a = max(0.0, min(a, 1000.0))
        return int(round(a / 1000.0 * total))

    x1 = norm1000_to_pixel(x1_n, width) - padding
    x2 = norm1000_to_pixel(x2_n, width) + padding
    y1 = norm1000_to_pixel(y1_n, height) - padding
    y2 = norm1000_to_pixel(y2_n, height) + padding

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    if (x2 - x1) < min_size:
        x1, x2 = _expand_to_min_size(x1, x2, width, min_size)
    if (y2 - y1) < min_size:
        y1, y2 = _expand_to_min_size(y1, y2, height, min_size)

    return x1, y1, x2, y2


def _map_crop_bbox_to_original(crop_bbox, crop_box, original_size):
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    original_width, original_height = original_size

    x1_n, y1_n, x2_n, y2_n = [max(0.0, min(float(v), 1000.0)) for v in crop_bbox["bbox_2d"]]
    x1 = crop_x1 + x1_n / 1000.0 * crop_width
    x2 = crop_x1 + x2_n / 1000.0 * crop_width
    y1 = crop_y1 + y1_n / 1000.0 * crop_height
    y2 = crop_y1 + y2_n / 1000.0 * crop_height

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    return {
        "bbox_2d": [
            x1 / original_width * 1000.0,
            y1 / original_height * 1000.0,
            x2 / original_width * 1000.0,
            y2 / original_height * 1000.0,
        ]
    }


def locate_refer_object_batch(image_inputs, refer_objects):
    results = [None] * len(refer_objects)
    direct_indices = []
    refinement_items = []

    for idx, refer_object in enumerate(refer_objects):
        relation = _extract_part_relation(refer_object)
        if relation is None:
            direct_indices.append(idx)
        else:
            parent_phrase, part_phrase = relation
            refinement_items.append((idx, parent_phrase, part_phrase))

    if direct_indices:
        direct_outputs = _locate_refer_object_batch_direct(
            [image_inputs[idx] for idx in direct_indices],
            [refer_objects[idx] for idx in direct_indices],
            prompt_mode="direct",
        )
        for idx, output in zip(direct_indices, direct_outputs):
            results[idx] = output

    if refinement_items:
        parent_outputs = _locate_refer_object_batch_direct(
            [image_inputs[idx] for idx, _, _ in refinement_items],
            [parent_phrase for _, parent_phrase, _ in refinement_items],
            prompt_mode="parent",
        )

        part_images = []
        part_phrases = []
        part_metadata = []
        for item, parent_bboxes in zip(refinement_items, parent_outputs):
            idx, _, part_phrase = item
            if not parent_bboxes:
                continue
            image = image_inputs[idx]
            crop_box = _bbox_norm_to_pixels(
                image,
                parent_bboxes[0],
                padding=PART_REFINEMENT_PADDING,
                min_size=PART_REFINEMENT_MIN_SIZE,
            )
            crop = image.crop(crop_box)
            part_images.append(crop)
            part_phrases.append(part_phrase)
            part_metadata.append((idx, crop_box, image.size))

        if part_images:
            part_outputs = _locate_refer_object_batch_direct(
                part_images,
                part_phrases,
                prompt_mode="part",
            )
            for metadata, part_bboxes in zip(part_metadata, part_outputs):
                idx, crop_box, original_size = metadata
                if part_bboxes:
                    results[idx] = [
                        _map_crop_bbox_to_original(part_bboxes[0], crop_box, original_size)
                    ]

    fallback_indices = [idx for idx, result in enumerate(results) if result is None]
    if fallback_indices:
        fallback_outputs = _locate_refer_object_batch_direct(
            [image_inputs[idx] for idx in fallback_indices],
            [refer_objects[idx] for idx in fallback_indices],
            prompt_mode="direct",
        )
        for idx, output in zip(fallback_indices, fallback_outputs):
            results[idx] = output

    return results


def locate_refer_object(image_input, refer_object: str):
    return locate_refer_object_batch([image_input], [refer_object])[0]


def crop_with_bbox(
    image_input,
    bbox,
    crop_dir: str,
    index: int = 0,
    padding: int = 0,
):
    x1_n, y1_n, x2_n, y2_n = [float(v) for v in bbox["bbox_2d"]]
    image = image_input
    width, height = image.size

    def norm1000_to_pixel(a, total):
        a = max(0.0, min(a, 1000.0))
        return int(round(a / 1000.0 * total))

    x1 = norm1000_to_pixel(x1_n, width)
    x2 = norm1000_to_pixel(x2_n, width)
    y1 = norm1000_to_pixel(y1_n, height)
    y2 = norm1000_to_pixel(y2_n, height)

    if padding > 0:
        x1 -= padding
        y1 -= padding
        x2 += padding
        y2 += padding

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    min_size = 64

    def _expand_to_min(a1: int, a2: int, limit: int, target: int):
        size = a2 - a1
        if size >= target:
            return a1, a2
        extra = target - size
        left = extra // 2
        right = extra - left
        a1 -= left
        a2 += right
        if a1 < 0:
            a2 += -a1
            a1 = 0
        if a2 > limit:
            a1 -= a2 - limit
            a2 = limit
            if a1 < 0:
                a1 = 0
        return a1, a2

    if (x2 - x1) < min_size:
        x1, x2 = _expand_to_min(x1, x2, width, min_size)
    if (y2 - y1) < min_size:
        y1, y2 = _expand_to_min(y1, y2, height, min_size)

    crop = image.crop((x1, y1, x2, y2))

    os.makedirs(crop_dir, exist_ok=True)
    save_name = f"crop_{index:02d}.png"
    save_path = os.path.join(crop_dir, save_name)
    crop.save(save_path)
    return save_path, (x1, y1, x2, y2)
