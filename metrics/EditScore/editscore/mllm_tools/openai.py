import base64
from io import BytesIO
from typing import Optional, Tuple, Union

import requests
from PIL import Image, ImageOps


def encode_pil_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_image(
    image: Union[str, Image.Image],
    mode: str = "RGB",
    size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            image = Image.open(requests.get(image, stream=True).raw)
        else:
            image = Image.open(image)
    image = ImageOps.exif_transpose(image).convert(mode)
    if size is not None:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


class GPT4v:
    def __init__(
        self,
        key,
        url="https://api.openai.com/v1/chat/completions",
        are_images_encoded=False,
        model_name="gpt-4-vision-preview",
    ):
        self.api_key = key
        self.url = url
        self.model_name = model_name
        self.use_encode = are_images_encoded

    def prepare_input(self, image_links=None, text_prompt: str = ""):
        image_links = [] if image_links is None else image_links
        if not isinstance(image_links, list):
            image_links = [image_links]

        content = [{"type": "text", "text": text_prompt}]
        for image_link in image_links:
            image = load_image(image_link)
            url = (
                f"data:image/jpeg;base64,{encode_pil_image(image)}"
                if self.use_encode
                else image_link
            )
            content.append({"type": "image_url", "image_url": {"url": url}})
        return content

    def inference(self, prompt, seed: Optional[int] = None):
        response = requests.post(
            self.url,
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=180,
        )
        return self.extract_response(response)

    def extract_response(self, response):
        data = response.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        error = data["error"]
        code = error["code"]
        print(error["message"], flush=True)
        if code in {
            "rate_limit_exceeded",
            "insufficient_quota",
            "insufficient_user_quota",
        }:
            return "rate_limit_exceeded"
        return ""


class GPT4o(GPT4v):
    def __init__(
        self,
        key,
        url="https://api.openai.com/v1/chat/completions",
        are_images_encoded=False,
        model_name="gpt-4o-2024-05-13",
    ):
        super().__init__(key, url, are_images_encoded, model_name)
