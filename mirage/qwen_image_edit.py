import copy

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)
from tqdm import tqdm

from .composition import compose
from .geometry import bbox_to_latent, crop_cells, min_size_box

# Latent cells of context kept around each box in its branch crop.
CONTEXT_CELLS = 2
# Qwen-Image-Edit works at a fixed resolution; a branch crop is upscaled at most this much per side.
MAX_UPSCALE = 4
# Width in latent cells of the soft ring around the boxes during the global stage.
BLEND_CELLS = 2


def _grid(tokens, latent_hw):
    return tokens.reshape(tokens.shape[0], *latent_hw, tokens.shape[-1]).permute(0, 3, 1, 2)


def _tokens(grid):
    batch, channels, height, width = grid.shape
    return grid.permute(0, 2, 3, 1).reshape(batch, height * width, channels)


def _blend_weight(latent_hw, boxes, device, dtype):
    """1 inside the boxes, fading linearly to 0 over BLEND_CELLS rings around them."""
    weight = torch.zeros(1, 1, *latent_hw, device=device)
    for y1, y2, x1, x2 in boxes:
        weight[..., y1:y2, x1:x2] = 1.0
    ring = weight
    for step in range(1, BLEND_CELLS + 1):
        ring = F.max_pool2d(ring, kernel_size=3, stride=1, padding=1)
        weight = torch.maximum(weight, ring * (1.0 - step / (BLEND_CELLS + 1)))
    return weight.to(dtype)


def _per_step_weight(target, steps):
    """Per-step blend weight w whose mean compounded effect (w + ... + w**steps) / steps hits target."""
    goal = target.float()
    low, high = torch.zeros_like(goal), torch.ones_like(goal)
    for _ in range(60):
        mid = 0.5 * (low + high)
        effect = mid * (1.0 - mid.pow(steps)) / (steps * (1.0 - mid).clamp(min=1e-12))
        below = effect < goal
        low = torch.where(below, mid, low)
        high = torch.where(below, high, mid)
    w = 0.5 * (low + high)
    w = torch.where(goal >= 1.0, 1.0, w)
    w = torch.where(goal <= 0.0, 0.0, w)
    return w.to(target.dtype)


def _forward(pipe, latents, branch, text, timestep):
    return pipe.transformer(
        hidden_states=torch.cat([latents, branch["cond"]], dim=1),
        timestep=timestep / 1000,
        encoder_hidden_states_mask=text[1],
        encoder_hidden_states=text[0],
        img_shapes=branch["img_shapes"],
        return_dict=False,
    )[0][:, : latents.shape[1]]


@torch.no_grad()
def run_qwen_image_edit(
    pipe,
    image,
    prompt,
    sub_prompts,
    bboxes,
    num_inference_steps=40,
    true_cfg_scale=4.0,
    patch_ratio=0.4,
    generator=None,
    negative_prompt=" ",
):
    """MIRAGE on Qwen-Image-Edit-2511; `bboxes` are pixel boxes of `image`."""
    device = pipe._execution_device
    dtype = pipe.transformer.dtype
    channels = pipe.transformer.config.in_channels // 4
    multiple_of = pipe.vae_scale_factor * 2
    use_cfg = true_cfg_scale > 1

    def encode(img, text, target_area):
        ratio = img.width / img.height
        width, height = calculate_dimensions(target_area, ratio)
        width, height = int(width) // multiple_of * multiple_of, int(height) // multiple_of * multiple_of
        cond_width, cond_height = calculate_dimensions(target_area * (CONDITION_IMAGE_SIZE / VAE_IMAGE_SIZE), ratio)
        cond_image = pipe.image_processor.resize(img, int(cond_height), int(cond_width))
        vae_image = pipe.image_processor.preprocess(img, height=height, width=width).unsqueeze(2)
        latents, cond = pipe.prepare_latents(
            images=[vae_image.to(device=device, dtype=pipe.vae.dtype)],
            batch_size=1,
            num_channels_latents=channels,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        def encode_text(t):
            embeds, mask = pipe.encode_prompt(image=[cond_image], prompt=t, device=device)
            return embeds.to(dtype), mask

        latent_hw = (height // multiple_of, width // multiple_of)
        return {
            "size": (width, height),
            "latent_hw": latent_hw,
            "latents": latents,
            "cond": cond,
            "text": encode_text(text),
            "negative": encode_text(negative_prompt) if use_cfg else None,
            "img_shapes": [[(1, *latent_hw), (1, *latent_hw)]],
        }

    def velocity(latents, branch, timestep):
        v = _forward(pipe, latents, branch, branch["text"], timestep)
        if branch["negative"] is None:
            return v
        v_neg = _forward(pipe, latents, branch, branch["negative"], timestep)
        combined = v_neg + true_cfg_scale * (v - v_neg)
        return combined * (torch.norm(v, dim=-1, keepdim=True) / torch.norm(combined, dim=-1, keepdim=True))

    full = encode(image, prompt, VAE_IMAGE_SIZE)
    latent_hw = full["latent_hw"]
    noise = _grid(full["latents"], latent_hw)
    clean = _grid(full["cond"], latent_hw)

    # Branches run on upscaled crops with their own noise; at the handover their clean
    # estimates are resampled to the canvas grid and re-noised with the global noise.
    branches = []
    for sub_prompt, bbox in zip(sub_prompts, bboxes):
        box = min_size_box(bbox_to_latent(bbox, image.size, latent_hw), full["size"], latent_hw)
        y1, y2, x1, x2 = box
        crop_box = (
            max(0, y1 - CONTEXT_CELLS),
            min(latent_hw[0], y2 + CONTEXT_CELLS),
            max(0, x1 - CONTEXT_CELLS),
            min(latent_hw[1], x2 + CONTEXT_CELLS),
        )
        crop = crop_cells(image, full["size"], latent_hw, crop_box)
        crop_area = (crop_box[1] - crop_box[0]) * (crop_box[3] - crop_box[2]) * multiple_of**2
        min_area = pipe.scheduler.config.base_image_seq_len * multiple_of**2
        branch = encode(crop, sub_prompt, min(VAE_IMAGE_SIZE, max(crop_area * MAX_UPSCALE**2, min_area)))
        branch["box"] = box
        branch["crop_box"] = crop_box
        branch["scheduler"] = copy.deepcopy(pipe.scheduler)
        branches.append(branch)

    scheduler = copy.deepcopy(pipe.scheduler)
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    config = scheduler.config

    def mu(seq_len):
        return calculate_shift(
            seq_len, config.base_image_seq_len, config.max_image_seq_len, config.base_shift, config.max_shift
        )

    timesteps, _ = retrieve_timesteps(
        scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu(full["latents"].shape[1])
    )
    for branch in branches:
        retrieve_timesteps(
            branch["scheduler"], num_inference_steps, device, sigmas=sigmas, mu=mu(branch["latents"].shape[1])
        )

    def noised(x0, eps, i):
        # forward diffusion to the noise level reached after step i
        sigma = scheduler.sigmas[i + 1].to(x0.dtype)
        return sigma * eps + (1 - sigma) * x0

    handover = int(len(timesteps) * patch_ratio)
    weight = _blend_weight(latent_hw, [branch["box"] for branch in branches], clean.device, clean.dtype)
    weight = _per_step_weight(weight, max(len(timesteps) - handover, 1))

    latents = full["latents"]
    for i, t in enumerate(tqdm(timesteps, leave=False)):
        timestep = t.expand(1).to(dtype)
        if i < handover - 1:
            for branch in branches:
                v = velocity(branch["latents"], branch, timestep)
                branch["latents"] = branch["scheduler"].step(v, t, branch["latents"], return_dict=False)[0]
        elif i == handover - 1:
            pasted = []
            for branch in branches:
                v = velocity(branch["latents"], branch, timestep)
                x0 = branch["latents"] - branch["scheduler"].sigmas[i].to(dtype) * v
                y1, y2, x1, x2 = branch["box"]
                cy1, cy2, cx1, cx2 = branch["crop_box"]
                x0 = F.adaptive_avg_pool2d(_grid(x0, branch["latent_hw"]).float(), (cy2 - cy1, cx2 - cx1))
                x0 = x0.to(dtype)[..., y1 - cy1 : y2 - cy1, x1 - cx1 : x2 - cx1]
                pasted.append((noised(x0, noise[..., y1:y2, x1:x2], i), branch["box"], x0 - clean[..., y1:y2, x1:x2]))
            latents = _tokens(compose(noised(clean, noise, i), pasted))
        else:
            v = velocity(latents, full, timestep)
            latents = scheduler.step(v, t, latents, return_dict=False)[0]
            canvas = noised(clean, noise, i)
            canvas = canvas + weight * (_grid(latents, latent_hw).to(canvas.dtype) - canvas)
            latents = _tokens(canvas)

    latents = pipe._unpack_latents(latents, full["size"][1], full["size"][0], pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)
    z_dim = pipe.vae.config.z_dim
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]
