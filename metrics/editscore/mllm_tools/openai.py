import base64
from io import BytesIO

import requests
from PIL import ImageOps


class GPT:
    def __init__(self, model_name, api_key, url="https://api.openai.com/v1/chat/completions"):
        self.model_name = model_name
        self.api_key = api_key
        self.url = url

    def prepare_input(self, images, text_prompt):
        content = [{"type": "text", "text": text_prompt}]
        for image in images:
            buffer = BytesIO()
            ImageOps.exif_transpose(image).convert("RGB").save(buffer, format="JPEG")
            url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": url}})
        return content

    def inference(self, content, seed=None):
        response = requests.post(
            self.url,
            json={"model": self.model_name, "messages": [{"role": "user", "content": content}]},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
