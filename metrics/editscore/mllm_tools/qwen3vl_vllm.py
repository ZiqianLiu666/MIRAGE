import hashlib
import os

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from vllm import LLM, SamplingParams


def merge_lora(model_name, lora_path, cache_dir=None):
    """Merge the LoRA weights once and cache the merged model, since vLLM loads full checkpoints."""
    if cache_dir is None:
        name = os.path.splitext(os.path.basename(lora_path))[0]
        digest = hashlib.md5(lora_path.encode()).hexdigest()[:8]
        cache_dir = os.path.join(
            torch.hub.get_dir(), "EditScore", f"{os.path.basename(model_name)}_merged_lora_{name}_{digest}"
        )
    if not os.path.exists(cache_dir):
        print(f"Merging {lora_path} into {model_name}, saving to {cache_dir}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cpu")
        PeftModel.from_pretrained(model, lora_path).merge_and_unload().save_pretrained(cache_dir)
        AutoProcessor.from_pretrained(model_name).save_pretrained(cache_dir)
    return cache_dir


class Qwen3VL:
    def __init__(
        self,
        model_name,
        temperature=0.7,
        lora_path=None,
        cache_dir=None,
        tensor_parallel_size=1,
        max_model_len=1536,
        max_num_seqs=32,
        max_num_batched_tokens=1536,
    ):
        if lora_path:
            model_name = merge_lora(model_name, lora_path, cache_dir)
        self.model = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            limit_mm_per_prompt={"image": 2},
            enable_prefix_caching=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.temperature = temperature

    def prepare_input(self, images, text_prompt):
        content = [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": text_prompt}]
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "multi_modal_data": {"image": process_vision_info(messages)[0]}}

    def inference(self, inputs, seed):
        params = SamplingParams(max_tokens=512, temperature=self.temperature, top_p=0.9, top_k=20, seed=seed)
        return self.model.generate(inputs, params, use_tqdm=False)[0].outputs[0].text.strip()
