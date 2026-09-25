import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed


class Qwen3VL:
    def __init__(self, model_name, temperature=0.7, lora_path=None):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto"
        )
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path).merge_and_unload()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.temperature = temperature

    def prepare_input(self, images, text_prompt):
        content = [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": text_prompt}]
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

    def inference(self, inputs, seed):
        set_seed(seed)
        output = self.model.generate(
            **inputs, max_new_tokens=512, do_sample=True, temperature=self.temperature, top_p=0.9, top_k=20
        )
        text = self.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return text[0].strip()
