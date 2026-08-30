import gc
import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

QWEN3_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
QWEN4_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
QWEN35_MODEL_ID = "Qwen/Qwen3.5-9B"
GEMMA4_MODEL_ID = "google/gemma-4-12B-it"
REGIONREASONER_MODEL_ID = "lmsdss/RegionReasoner-7B"
REGIONREASONER_IMAGE_SIZE = 840

QWEN_MIN_VISUAL_TOKENS = 768
QWEN_MAX_VISUAL_TOKENS = 3072
QWEN3_VL_SPATIAL_COMPRESSION = 32

GEMMA4_VISUAL_TOKEN_CHOICES = (70, 140, 280, 560, 1120)
GEMMA4_DEFAULT_VISUAL_TOKENS = 1120

PART_REFINEMENT_PADDING = 32
PART_REFINEMENT_MIN_SIZE = 192
PART_REFINEMENT_AREA_RATIO = 0.6
PART_REFINEMENT_CONTAINMENT = 0.5
PART_MAX_WORDS = 8


def _dtype_from_name(name: str):
    table = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return table[name.lower()]


@dataclass(frozen=True)
class _BackendConfig:
    name: str = "gemma4"
    model_id: Optional[str] = None
    device: str = "cuda:0"
    dtype: str = "bf16"
    gemma_visual_tokens: int = GEMMA4_DEFAULT_VISUAL_TOKENS


_backend_config = _BackendConfig()
_backend = None


def configure_backend(
    name: str = "gemma4",
    model_id: Optional[str] = None,
    device: str = "cuda:0",
    dtype: str = "bf16",
    gemma_visual_tokens: int = GEMMA4_DEFAULT_VISUAL_TOKENS,
):
    global _backend_config, _backend

    name = name.lower().strip()
    name = {
        "qwen": "qwen3",
        "qwen8b": "qwen3",
        "qwen4b": "qwen4",
    }.get(name, name)
    new_config = _BackendConfig(
        name=name,
        model_id=model_id,
        device=device,
        dtype=dtype,
        gemma_visual_tokens=gemma_visual_tokens,
    )
    if new_config == _backend_config:
        return

    if _backend is not None:
        _backend = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _backend_config = new_config


class _Qwen3Backend:
    coordinate_order = "xyxy"
    grounding_max_new_tokens = 128

    def __init__(
        self,
        config: _BackendConfig,
        default_model_id: str = QWEN3_MODEL_ID,
    ):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = torch.device(config.device)
        self.dtype = _dtype_from_name(config.dtype)
        self.model_id = config.model_id or default_model_id

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        )
        self.model = self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.processor.tokenizer.padding_side = "left"
        self.processor.image_processor.size = {
            "shortest_edge": QWEN_MIN_VISUAL_TOKENS
            * QWEN3_VL_SPATIAL_COMPRESSION
            * QWEN3_VL_SPATIAL_COMPRESSION,
            "longest_edge": QWEN_MAX_VISUAL_TOKENS
            * QWEN3_VL_SPATIAL_COMPRESSION
            * QWEN3_VL_SPATIAL_COMPRESSION,
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

    @staticmethod
    def bbox_output_instructions() -> str:
        return """
        Coordinate system and output:
        - Use NORMALIZED coordinates in the range [0, 1000].
        - 0 = left/top image edge, 1000 = right/bottom image edge.
        - Output integers only. Do NOT output pixel coordinates or percentages.
        - Return ONLY ONE valid JSON array and NOTHING ELSE.
        - JSON format (must match EXACTLY):
          [{"bbox_2d": [x_min, y_min, x_max, y_max], "label": "object"}]
        """.strip()

    @staticmethod
    def parse_bboxes(text: str):
        data = _load_json_from_model_text(text)
        return _extract_bboxes(
            data,
            key_order={
                "bbox_2d": "xyxy",
                "box_2d": "xyxy",
                "bbox": "xyxy",
                "box": "xyxy",
                "bounding_box": "xyxy",
            },
            default_order="xyxy",
        )


class _RegionReasonerBackend:

    grounding_max_new_tokens = 2048
    resize_size = REGIONREASONER_IMAGE_SIZE
    detection_template = (
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

    def __init__(self, config: _BackendConfig):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = torch.device(config.device)
        self.dtype = _dtype_from_name(config.dtype)
        self.model_id = config.model_id or REGIONREASONER_MODEL_ID
        self._process_vision_info = process_vision_info

        load_kwargs = {"torch_dtype": self.dtype}
        if self.device.type == "cuda":
            load_kwargs["device_map"] = {"": str(self.device)}

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                attn_implementation="flash_attention_2",
                **load_kwargs,
            )
        except (ImportError, RuntimeError, ValueError):
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **load_kwargs,
            )
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            padding_side="left",
        )

    def chat_batch(self, batch_messages, max_new_tokens=512):
        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in batch_messages
        ]
        inputs = self.processor(
            text=texts,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_token_id,
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

    @classmethod
    def parse_bboxes(cls, output_text: str):
        candidates = []
        answer_match = re.search(
            r"<answer>\s*(.*?)\s*</answer>",
            str(output_text),
            flags=re.DOTALL,
        )
        if answer_match:
            candidates.append(answer_match.group(1).strip())
        candidates.append(str(output_text).strip())

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                try:
                    data = _load_json_from_model_text(candidate)
                except ValueError:
                    continue
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                continue

            boxes = []
            for item in data:
                raw_bbox = item.get("bbox_2d") if isinstance(item, dict) else None
                if not _is_bbox_quad(raw_bbox):
                    continue
                x1, y1, x2, y2 = [float(v) for v in raw_bbox]
                x1, x2 = sorted((x1, x2))
                y1, y2 = sorted((y1, y2))
                scale = 1000.0 / cls.resize_size
                boxes.append(
                    {
                        "bbox_2d": [
                            max(0.0, min(1000.0, x1 * scale)),
                            max(0.0, min(1000.0, y1 * scale)),
                            max(0.0, min(1000.0, x2 * scale)),
                            max(0.0, min(1000.0, y2 * scale)),
                        ]
                    }
                )
            if boxes:
                return boxes
        return []

    def locate_batch(self, images, queries):
        batch_messages = []
        for image, query in zip(images, queries):
            resized_image = image.convert("RGB").resize(
                (self.resize_size, self.resize_size),
                Image.Resampling.BILINEAR,
            )
            prompt = self.detection_template.format(
                Question=str(query).lower().strip('."?!'),
                Answer='[{"bbox_2d": [10,100,200,210], "point_2d": [30,110]}]',
            )
            batch_messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": resized_image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )

        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in batch_messages
        ]
        image_inputs, _ = self._process_vision_info(batch_messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=self.grounding_max_new_tokens,
                do_sample=False,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [self.parse_bboxes(text) for text in output_texts]


class _Qwen35Backend:

    coordinate_order = "xyxy"
    grounding_max_new_tokens = 128

    def __init__(self, config: _BackendConfig):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.device = torch.device(config.device)
        self.dtype = _dtype_from_name(config.dtype)
        self.model_id = config.model_id or QWEN35_MODEL_ID

        load_kwargs = {
            "dtype": self.dtype,
            "attn_implementation": "sdpa",
        }
        if self.device.type == "cuda":
            load_kwargs["device_map"] = {"": str(self.device)}

        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            **load_kwargs,
        )
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            padding_side="left",
        )

    def chat_batch(self, batch_messages, max_new_tokens=512):
        inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
            processor_kwargs={"padding": True},
        ).to(self.device)

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_token_id,
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

    @staticmethod
    def bbox_output_instructions() -> str:
        return """
        Coordinate system and output:
        - Use NORMALIZED coordinates in the range [0, 1000].
        - 0 = left/top image edge, 1000 = right/bottom image edge.
        - Return coordinates in this exact order: [x_min, y_min, x_max, y_max].
        - Output integers only. Do NOT output pixel coordinates or percentages.
        - Return ONLY ONE valid JSON array and NOTHING ELSE.
        - JSON format (must match EXACTLY):
          [{"bbox_2d": [x_min, y_min, x_max, y_max], "label": "object"}]
        """.strip()

    @staticmethod
    def parse_bboxes(text: str):
        data = _load_json_from_model_text(text)

        bbox_keys = ("bbox_2d", "bbox", "box", "bounding_box", "box_2d")
        wrapper_keys = ("objects", "detections", "results", "boxes", "bboxes")

        def is_number(v):
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        def is_bbox(v):
            return (
                isinstance(v, (list, tuple))
                and len(v) == 4
                and all(is_number(x) for x in v)
            )

        def make_bbox(values):
            x1, y1, x2, y2 = [float(v) for v in values]
            vals = [
                max(0.0, min(1000.0, min(x1, x2))),
                max(0.0, min(1000.0, min(y1, y2))),
                max(0.0, min(1000.0, max(x1, x2))),
                max(0.0, min(1000.0, max(y1, y2))),
            ]
            return {"bbox_2d": vals}

        def extract(obj):
            if isinstance(obj, dict):
                for key in bbox_keys:
                    if key in obj:
                        value = obj[key]
                        if isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except json.JSONDecodeError:
                                return []
                        if is_bbox(value):
                            return [make_bbox(value)]
                for key in wrapper_keys:
                    if key in obj:
                        return extract(obj[key])
                return []

            if not isinstance(obj, (list, tuple)):
                return []

            if is_bbox(obj):
                return [make_bbox(obj)]

            if len(obj) == 2:
                if is_bbox(obj[0]) and isinstance(obj[1], str):
                    return [make_bbox(obj[0])]
                if isinstance(obj[0], str) and is_bbox(obj[1]):
                    return [make_bbox(obj[1])]
            if len(obj) == 1 and is_bbox(obj[0]):
                return [make_bbox(obj[0])]

            out = []
            for item in obj:
                if isinstance(item, str):
                    continue
                out.extend(extract(item))
            return out

        return extract(data)

def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_bbox_quad(v):
    return isinstance(v, (list, tuple)) and len(v) == 4 and all(_is_number(x) for x in v)


def _extract_bboxes(data, key_order, default_order="xyxy"):
    wrapper_keys = ("objects", "detections", "results", "boxes", "bboxes")

    def to_bbox_2d(values, order):
        a, b, c, d = [float(v) for v in values]
        if order == "yxyx":
            y1, x1, y2, x2 = a, b, c, d
        else:
            x1, y1, x2, y2 = a, b, c, d
        return {
            "bbox_2d": [
                max(0.0, min(1000.0, min(x1, x2))),
                max(0.0, min(1000.0, min(y1, y2))),
                max(0.0, min(1000.0, max(x1, x2))),
                max(0.0, min(1000.0, max(y1, y2))),
            ]
        }

    def extract(obj):
        if isinstance(obj, dict):
            for key, order in key_order.items():
                if key in obj:
                    value = obj[key]
                    if isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            return []
                    if _is_bbox_quad(value):
                        return [to_bbox_2d(value, order)]
            for key in wrapper_keys:
                if key in obj:
                    return extract(obj[key])
            return []

        if not isinstance(obj, (list, tuple)):
            return []

        if _is_bbox_quad(obj):
            return [to_bbox_2d(obj, default_order)]

        if len(obj) == 2:
            if _is_bbox_quad(obj[0]) and isinstance(obj[1], str):
                return [to_bbox_2d(obj[0], default_order)]
            if isinstance(obj[0], str) and _is_bbox_quad(obj[1]):
                return [to_bbox_2d(obj[1], default_order)]
        if len(obj) == 1 and _is_bbox_quad(obj[0]):
            return [to_bbox_2d(obj[0], default_order)]

        out = []
        for item in obj:
            if isinstance(item, str):
                continue
            out.extend(extract(item))
        return out

    return extract(data)


class _Gemma4Backend:
    coordinate_order = "yx_yx"
    grounding_max_new_tokens = 256

    def __init__(self, config: _BackendConfig):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.device = torch.device(config.device)
        self.dtype = _dtype_from_name(config.dtype)
        self.model_id = config.model_id or GEMMA4_MODEL_ID
        self.visual_tokens = config.gemma_visual_tokens

        load_kwargs = {
            "dtype": self.dtype,
            "attn_implementation": "sdpa",
        }
        if self.device.type == "cuda":
            load_kwargs["device_map"] = {"": str(self.device)}

        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            **load_kwargs,
        )
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            padding_side="left",
        )
        self.processor.tokenizer.padding_side = "left"
        self.processor.image_processor.max_soft_tokens = self.visual_tokens

    def chat_batch(self, batch_messages, max_new_tokens=512):
        inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
            processor_kwargs={"padding": True},
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

    @staticmethod
    def bbox_output_instructions() -> str:
        return """
        Coordinate system and output:
        - Use Gemma's native normalized 1000x1000 object-detection coordinates.
        - Return coordinates in this exact order: [y_min, x_min, y_max, x_max].
        - Output integers only. Do NOT output pixel coordinates or percentages.
        - Return ONLY ONE valid JSON array and NOTHING ELSE.
        - JSON format (must match EXACTLY):
          [{"box_2d": [y_min, x_min, y_max, x_max], "label": "object"}]
        """.strip()

    @staticmethod
    def parse_bboxes(text: str):
        data = _load_json_from_model_text(text)
        return _extract_bboxes(
            data,
            key_order={"box_2d": "yxyx", "bbox_2d": "xyxy"},
            default_order="yxyx",
        )


def get_backend():
    global _backend
    if _backend is None:
        if _backend_config.name == "gemma4":
            _backend = _Gemma4Backend(_backend_config)
        elif _backend_config.name == "qwen3":
            _backend = _Qwen3Backend(_backend_config)
        elif _backend_config.name == "qwen4":
            _backend = _Qwen3Backend(
                _backend_config,
                default_model_id=QWEN4_MODEL_ID,
            )
        elif _backend_config.name == "qwen35":
            _backend = _Qwen35Backend(_backend_config)
        elif _backend_config.name == "regionreasoner":
            _backend = _RegionReasonerBackend(_backend_config)
        display_name = {
            "qwen3": "qwen8b",
            "qwen4": "qwen4b",
        }.get(_backend_config.name, _backend_config.name)
        print(
            f"Loaded VLM backend={display_name} "
            f"model={_backend.model_id} device={_backend_config.device} "
            f"dtype={_backend_config.dtype}"
        )
    return _backend


_JSON_DECODER = json.JSONDecoder()


def _load_json_from_model_text(text: str):
    cleaned = str(text).strip()
    starts = [pos for pos in (cleaned.find("["), cleaned.find("{")) if pos >= 0]
    if not starts:
        return []
    start = min(starts)

    data, _end = _JSON_DECODER.raw_decode(cleaned, start)
    return data

def _parse_items_from_text(text: str):
    data = _load_json_from_model_text(text)
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


def _clean_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().strip(".")).strip()


def _strip_leading_article(text: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", _clean_phrase(text), flags=re.IGNORECASE)


def _is_plausible_part_split(parent: str, part: str) -> bool:
    if not parent or not part:
        return False
    if len(part.split()) > PART_MAX_WORDS:
        return False
    return True


def _extract_part_relation(refer_object: str):
    phrase = _clean_phrase(refer_object)

    possessive = re.match(r"^(?P<parent>.+?)'s\s+(?P<part>.+)$", phrase)
    if possessive:
        parent = _strip_leading_article(possessive.group("parent"))
        part = _strip_leading_article(possessive.group("part"))
        if _is_plausible_part_split(parent, part):
            return parent, part

    of_relation = re.match(
        r"^(?:the|a|an)?\s*(?P<part>.+?)\s+of\s+(?P<parent>.+)$",
        phrase,
        flags=re.IGNORECASE,
    )
    if of_relation:
        parent = _strip_leading_article(of_relation.group("parent"))
        part = _strip_leading_article(of_relation.group("part"))
        if _is_plausible_part_split(parent, part):
            return parent, part

    return None


def _bbox_xyxy(bbox) -> tuple:
    x1, y1, x2, y2 = [float(v) for v in bbox["bbox_2d"]]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return x1, y1, x2, y2


def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_intersection_area(a, b) -> float:
    ax1, ay1, ax2, ay2 = _bbox_xyxy(a)
    bx1, by1, bx2, by2 = _bbox_xyxy(b)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    return inter_w * inter_h


def _needs_part_refinement(direct_bbox, parent_bbox) -> bool:
    if direct_bbox is None or parent_bbox is None:
        return True

    parent_area = _bbox_area(parent_bbox)
    direct_area = _bbox_area(direct_bbox)
    if parent_area <= 0.0 or direct_area <= 0.0:
        return True

    if direct_area / parent_area >= PART_REFINEMENT_AREA_RATIO:
        return True

    containment = _bbox_intersection_area(direct_bbox, parent_bbox) / direct_area
    if containment < PART_REFINEMENT_CONTAINMENT:
        return True

    return False


def _build_qwen35_locate_prompt(refer_object: str, prompt_mode: str = "direct") -> str:
    if prompt_mode == "parent":
        target = (
            f"the full visible parent instance described by '{refer_object}'. "
            "Preserve spatial qualifiers and box the whole parent, not a small part"
        )
    elif prompt_mode == "part":
        target = (
            f"the visible part '{refer_object}' inside this cropped parent image. "
            "Box only that part; if it is a visible pair/group, box the pair/group together"
        )
    else:
        target = (
            f"the single visible target described by '{refer_object}'. "
            "If the phrase is 'X of Y' or Y\'s X, locate X only and use Y only "
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

        {get_backend().bbox_output_instructions()}

        STRICT JSON RULES:
        - Use double quotes "..." around all keys and string values.
        - Use colon ":" between keys and values.
        - No trailing commas.

        Target phrase: {refer_object}
        """.strip()


def _locate_refer_object_batch_direct(image_inputs, refer_objects, prompt_mode: str = "direct"):
    backend = get_backend()
    if _backend_config.name == "regionreasoner":
        return backend.locate_batch(image_inputs, refer_objects)

    batch_messages = []
    for image_input, refer_object in zip(image_inputs, refer_objects):
        if _backend_config.name == "qwen35":
            user_prompt = _build_qwen35_locate_prompt(
                refer_object, prompt_mode=prompt_mode
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_input},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
        else:
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
                },
            ]
        batch_messages.append(messages)

    outputs = backend.chat_batch(batch_messages, max_new_tokens=backend.grounding_max_new_tokens)
    return [backend.parse_bboxes(text) for text in outputs]


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
    if _backend_config.name == "regionreasoner":
        return get_backend().locate_batch(image_inputs, refer_objects)

    results = [None] * len(refer_objects)

    direct_outputs = _locate_refer_object_batch_direct(
        image_inputs,
        refer_objects,
        prompt_mode="direct",
    )
    for idx, output in enumerate(direct_outputs):
        results[idx] = output

    relation_items = []
    for idx, refer_object in enumerate(refer_objects):
        relation = _extract_part_relation(refer_object)
        if relation is not None:
            parent_phrase, part_phrase = relation
            relation_items.append((idx, parent_phrase, part_phrase))

    if not relation_items:
        return results

    parent_outputs = _locate_refer_object_batch_direct(
        [image_inputs[idx] for idx, _, _ in relation_items],
        [parent_phrase for _, parent_phrase, _ in relation_items],
        prompt_mode="parent",
    )

    part_images = []
    part_phrases = []
    part_metadata = []
    for (idx, _, part_phrase), parent_bboxes in zip(relation_items, parent_outputs):
        parent_bbox = parent_bboxes[0] if parent_bboxes else None
        direct_bbox = results[idx][0] if results[idx] else None

        if parent_bbox is None:
            continue
        if not _needs_part_refinement(direct_bbox, parent_bbox):
            continue

        image = image_inputs[idx]
        crop_box = _bbox_norm_to_pixels(
            image,
            parent_bbox,
            padding=PART_REFINEMENT_PADDING,
            min_size=PART_REFINEMENT_MIN_SIZE,
        )
        part_images.append(image.crop(crop_box))
        part_phrases.append(part_phrase)
        part_metadata.append((idx, crop_box, image.size, parent_bboxes))

    if part_images:
        part_outputs = _locate_refer_object_batch_direct(
            part_images,
            part_phrases,
            prompt_mode="part",
        )
        for metadata, part_bboxes in zip(part_metadata, part_outputs):
            idx, crop_box, original_size, parent_bboxes = metadata
            if part_bboxes:
                results[idx] = [
                    _map_crop_bbox_to_original(part_bboxes[0], crop_box, original_size)
                ]
            elif not results[idx]:
                results[idx] = parent_bboxes

    for (idx, _, _), parent_bboxes in zip(relation_items, parent_outputs):
        if not results[idx] and parent_bboxes:
            results[idx] = parent_bboxes

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
    image = image_input
    x1, y1, x2, y2 = _bbox_norm_to_pixels(image, bbox, padding=padding)

    os.makedirs(crop_dir, exist_ok=True)
    save_path = os.path.join(crop_dir, f"crop_{index:02d}.png")
    image.crop((x1, y1, x2, y2)).save(save_path)
    return save_path, (x1, y1, x2, y2)
