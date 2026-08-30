from typing import Optional
import random
import numpy as np
import torch

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Qwen3VL:
    def __init__(
        self,
        vlm_model,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        lora_path: Optional[str] = None,
    ) -> None:
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            vlm_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()

        self.processor = AutoProcessor.from_pretrained(vlm_model)
        self.temperature = temperature
        self.seed = seed

    def prepare_input(self, images, text_prompt: str = ""):
        if not isinstance(images, list):
            images = [images]

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": text_prompt}],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = inputs.to("cuda")

        return inputs

    def inference(self, inputs, seed: Optional[int] = None):
        seed = self.seed if seed is None else seed

        set_seed(seed)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.9,
            top_k=20,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        outputs = [output.strip() for output in outputs]
        return outputs[0]
