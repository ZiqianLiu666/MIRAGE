import random

import numpy as np
import torch

from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl.constants import IMAGE_TOKEN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_chat_template(prompt, num_images: int = 2):
    template = "\n".join([f"Image-{i}: {IMAGE_TOKEN}" for i in range(1, num_images + 1)])
    template += f"\n{prompt}"
    return template

class InternVL35:
    def __init__(self, model, max_model_len: int = 16384, tensor_parallel_size=1, max_num_seqs=32) -> None:
        self.model = pipeline(model, backend_config=PytorchEngineConfig(session_len=max_model_len, tp=tensor_parallel_size))

    def prepare_input(self, images=None, text_prompt: str = ""):
        images = [] if images is None else images
        if not isinstance(images, list):
            images = [images]
        messages = (apply_chat_template(text_prompt, num_images=len(images)), images)
        return messages

    def inference(self, messages):
        set_seed(42)
        response = self.model(messages)
        print(f"{response.text=}", flush=True)
        return response.text
