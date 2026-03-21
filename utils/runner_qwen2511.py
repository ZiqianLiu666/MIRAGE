import copy
import math
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from tqdm import tqdm


from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    calculate_shift,
    retrieve_timesteps,
)

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
        calculate_dimensions,
    )

from .geometry import bbox_to_latent_coords, coerce_bbox

from diffusers.utils.torch_utils import randn_tensor

def _get_device(pipe):
    if hasattr(pipe, "_execution_device") and pipe._execution_device is not None:
        return pipe._execution_device
    if hasattr(pipe, "device"):
        return pipe.device
    return pipe.transformer.device


def _resolve_qwen_condition_size(image: PIL.Image.Image) -> Tuple[int, int]:
    image_width, image_height = image.size
    ratio = image_width / image_height
    condition_width, condition_height = calculate_dimensions(CONDITION_IMAGE_SIZE, ratio)
    return int(condition_width), int(condition_height)


def _prepare_branch_state(
    pipe,
    image: PIL.Image.Image,
    prompt: str,
    negative_prompt: Optional[Union[str, List[str]]],
    do_true_cfg: bool,
    device: torch.device,
    latent_dtype: torch.dtype,
    generator: Optional[torch.Generator],
):
    image = image.convert("RGB")
    image_w, image_h = image.size
    image_ratio = image_w / image_h

    # Official __call__ logic: default width/height are derived from ~1MP area,
    # then snapped to vae_scale_factor * 2 multiples.
    width, height = calculate_dimensions(1024 * 1024, image_ratio)
    width = int(width)
    height = int(height)
    multiple_of = pipe.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    condition_w, condition_h = _resolve_qwen_condition_size(image)
    condition_image = pipe.image_processor.resize(image, condition_h, condition_w)

    # PR #12453 behavior: VAE path uses (width, height) directly.
    vae_tensor = pipe.image_processor.preprocess(image, height=height, width=width)
    if vae_tensor.ndim == 3:
        vae_tensor = vae_tensor.unsqueeze(0)
    vae_tensor = vae_tensor.unsqueeze(2)
    vae_tensor = vae_tensor.to(device=device, dtype=pipe.vae.dtype)

    prompt_embeds, prompt_mask = pipe.encode_prompt(
        image=[condition_image],
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
    )
    prompt_embeds = prompt_embeds.to(dtype=latent_dtype)

    negative_prompt_embeds = None
    negative_prompt_mask = None
    if do_true_cfg:
        negative_prompt_embeds, negative_prompt_mask = pipe.encode_prompt(
            image=[condition_image],
            prompt=negative_prompt,
            device=device,
            num_images_per_prompt=1,
        )
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=latent_dtype)

    latent_h = max(1, height // pipe.vae_scale_factor // 2)
    latent_w = max(1, width // pipe.vae_scale_factor // 2)
    img_shapes = [[(1, latent_h, latent_w), (1, latent_h, latent_w)]]

    return {
        "input_size": (image_w, image_h),
        "height": height,
        "width": width,
        "vae_tensor": vae_tensor,
        "prompt_embeds": prompt_embeds,
        "prompt_mask": prompt_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_mask": negative_prompt_mask,
        "img_shapes": img_shapes,
    }


def _make_noise_like(latents: torch.Tensor, generator: Optional[torch.Generator]):
    if generator is None:
        return torch.randn_like(latents)
    return randn_tensor(
        latents.shape, generator=generator, device=latents.device, dtype=latents.dtype
    )


def _add_noise_like(
    scheduler,
    sample: torch.Tensor,
    noise: torch.Tensor,
    timestep: Union[int, float, torch.Tensor],
):
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor(timestep, device=sample.device)
    else:
        timestep = timestep.to(sample.device)
    if timestep.ndim == 0:
        timestep = timestep.expand(sample.shape[0])
    return scheduler.scale_noise(sample=sample, timestep=timestep, noise=noise)


def _unpack_latents(pipe, latents: torch.Tensor, height: int, width: int):
    return pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)


def _pack_latents(pipe, latents: torch.Tensor):
    return pipe._pack_latents(
        latents,
        latents.shape[0],
        latents.shape[1],
        latents.shape[3],
        latents.shape[4],
    )


def _resize_latents(latents: torch.Tensor, target_h: int, target_w: int):
    _, _, t, h, w = latents.shape
    if h == target_h and w == target_w:
        return latents
    latents_4d = latents.reshape(latents.shape[0], latents.shape[1] * t, h, w)
    latents_4d = F.interpolate(
        latents_4d, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    return latents_4d.view(latents.shape[0], latents.shape[1], t, target_h, target_w)


def _predict_noise(
    pipe,
    latent_model_input: torch.Tensor,
    timestep: torch.Tensor,
    guidance: Optional[torch.Tensor],
    prompt_embeds: torch.Tensor,
    prompt_mask: Optional[torch.Tensor],
    img_shapes: List[List[Tuple[int, int, int]]],
    do_true_cfg: bool,
    true_cfg_scale: float,
    negative_prompt_embeds: Optional[torch.Tensor],
    negative_prompt_mask: Optional[torch.Tensor],
    latent_token_count: int,
):
    cond_context = (
        pipe.transformer.cache_context("cond")
        if hasattr(pipe.transformer, "cache_context")
        else nullcontext()
    )
    with cond_context:
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]
    noise_pred = noise_pred[:, :latent_token_count]

    if not do_true_cfg:
        return noise_pred

    uncond_context = (
        pipe.transformer.cache_context("uncond")
        if hasattr(pipe.transformer, "cache_context")
        else nullcontext()
    )
    with uncond_context:
        neg_noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states_mask=negative_prompt_mask,
            encoder_hidden_states=negative_prompt_embeds,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]
    neg_noise_pred = neg_noise_pred[:, :latent_token_count]

    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    return comb_pred * (cond_norm / noise_norm)


def _group_crop_state_indices(crop_states: List[Dict[str, Any]]):
    grouped = defaultdict(list)
    for idx, state in enumerate(crop_states):
        key = (
            state["latents"].shape[1],
            state["cond_latents"].shape[1],
            state["height"],
            state["width"],
            tuple(state["prompt_embeds"].shape),
            tuple(state["prompt_mask"].shape)
            if state["prompt_mask"] is not None
            else None,
        )
        grouped[key].append(idx)
    return list(grouped.values())


def _merge_batch_or_shared_tensors(tensors: List[Optional[torch.Tensor]]):
    if not tensors:
        return None
    first = tensors[0]
    if first is None:
        return None

    if first.ndim >= 1 and first.shape[0] == 1:
        return torch.cat(tensors, dim=0)
    return first


def _merge_img_shapes(group_states: List[Dict[str, Any]]):
    merged = []
    for state in group_states:
        merged.extend(state["img_shapes"])
    return merged


@torch.no_grad()
def run_qwen_multi_branch(
    pipe,
    full_image: PIL.Image.Image,
    crop_images: List[PIL.Image.Image],
    full_prompt: str,
    crop_prompts: List[str],
    bboxes: List[Optional[Union[Dict[str, int], Tuple[int, int, int, int], List[int]]]],
    num_inference_steps: int = 40,
    true_cfg_scale: float = 4.0,
    guidance_scale: Optional[float] = 1.0,
    negative_prompt: Optional[Union[str, List[str]]] = " ",
    generator: Optional[torch.Generator] = None,
    patch_ratio: float = 0.8,
):

    device = _get_device(pipe)
    latent_dtype = pipe.transformer.dtype

    has_neg_prompt = negative_prompt is not None
    do_true_cfg = true_cfg_scale > 1.0 and has_neg_prompt

    guidance = None
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)

    full_state = _prepare_branch_state(
        pipe=pipe,
        image=full_image,
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        do_true_cfg=do_true_cfg,
        device=device,
        latent_dtype=latent_dtype,
        generator=generator,
    )

    crop_states = []
    for crop_image, crop_prompt in zip(crop_images, crop_prompts):
        state = _prepare_branch_state(
            pipe=pipe,
            image=crop_image,
            prompt=crop_prompt,
            negative_prompt=negative_prompt,
            do_true_cfg=do_true_cfg,
            device=device,
            latent_dtype=latent_dtype,
            generator=generator,
        )
        crop_states.append(state)

    full_scheduler = copy.deepcopy(pipe.scheduler)
    crop_schedulers = [copy.deepcopy(pipe.scheduler) for _ in crop_states]

    num_channels_latents = pipe.transformer.config.in_channels // 4
    full_latents, full_cond_latents = pipe.prepare_latents(
        images=[full_state["vae_tensor"]],
        batch_size=1,
        num_channels_latents=num_channels_latents,
        height=full_state["height"],
        width=full_state["width"],
        dtype=latent_dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    full_latents = full_latents.to(device=device, dtype=latent_dtype)
    full_state["cond_latents"] = full_cond_latents.to(device=device, dtype=latent_dtype)

    for state in crop_states:
        crop_latents, crop_cond_latents = pipe.prepare_latents(
            images=[state["vae_tensor"]],
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=state["height"],
            width=state["width"],
            dtype=latent_dtype,
            device=device,
            generator=generator,
            latents=None,
        )
        state["latents"] = crop_latents.to(device=device, dtype=latent_dtype)
        state["cond_latents"] = crop_cond_latents.to(device=device, dtype=latent_dtype)

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = full_latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        full_scheduler.config.get("base_image_seq_len", 256),
        full_scheduler.config.get("max_image_seq_len", 4096),
        full_scheduler.config.get("base_shift", 0.5),
        full_scheduler.config.get("max_shift", 1.15),
    )

    timesteps, num_inference_steps = retrieve_timesteps(
        full_scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    for sch in crop_schedulers:
        retrieve_timesteps(sch, num_inference_steps, device, sigmas=sigmas, mu=mu)
    num_steps = len(timesteps)
    patch_ratio = float(patch_ratio)
    patch_ratio = max(0.0, min(patch_ratio, 1.0))
    patch_until = int(num_steps * patch_ratio)

    full_noise = _make_noise_like(full_state["cond_latents"], generator)

    for i, state in enumerate(crop_states):
        state["scheduler"] = crop_schedulers[i]
        state["noise"] = _make_noise_like(state["cond_latents"], generator)

    full_latents_img = _unpack_latents(
        pipe,
        full_state["cond_latents"],
        full_state["height"],
        full_state["width"],
    )
    full_latent_hw = (full_latents_img.shape[-2], full_latents_img.shape[-1])

    for i, state in enumerate(crop_states):
        latent_bbox = None
        if bboxes[i] is not None:
            latent_bbox = bbox_to_latent_coords(
                coerce_bbox(bboxes[i]),
                full_state["input_size"],
                full_latent_hw,
            )
        state["latent_bbox"] = latent_bbox

    any_bbox = any(state["latent_bbox"] is not None for state in crop_states)

    if guidance is not None:
        guidance = guidance.expand(full_latents.shape[0])

    for step_index, t in enumerate(
        tqdm(timesteps, desc="Diffusion steps", leave=False)
    ):
        base_full = None
        base_full_img = None
        if any_bbox:
            base_full = _add_noise_like(
                full_scheduler, full_state["cond_latents"], full_noise, t
            )
            base_full_img = _unpack_latents(
                pipe,
                base_full,
                full_state["height"],
                full_state["width"],
            )

        timestep = t.expand(full_latents.shape[0]).to(full_latents.dtype)

        if step_index < patch_until:
            for group_indices in _group_crop_state_indices(crop_states):
                group_states = [crop_states[i] for i in group_indices]
                batch_sizes = [state["latents"].shape[0] for state in group_states]

                latents_batch = torch.cat(
                    [state["latents"] for state in group_states], dim=0
                )
                cond_latents_batch = torch.cat(
                    [state["cond_latents"] for state in group_states], dim=0
                )
                crop_latent_model_input = torch.cat(
                    [latents_batch, cond_latents_batch], dim=1
                )

                timestep_batch = t.expand(latents_batch.shape[0]).to(
                    latents_batch.dtype
                )
                guidance_batch = (
                    guidance.expand(latents_batch.shape[0])
                    if guidance is not None
                    else None
                )
                prompt_embeds_batch = torch.cat(
                    [state["prompt_embeds"] for state in group_states], dim=0
                )
                prompt_mask_batch = _merge_batch_or_shared_tensors(
                    [state["prompt_mask"] for state in group_states]
                )
                img_shapes_batch = _merge_img_shapes(group_states)

                negative_prompt_embeds_batch = None
                negative_prompt_mask_batch = None
                if do_true_cfg:
                    negative_prompt_embeds_batch = _merge_batch_or_shared_tensors(
                        [state["negative_prompt_embeds"] for state in group_states]
                    )
                    negative_prompt_mask_batch = _merge_batch_or_shared_tensors(
                        [state["negative_prompt_mask"] for state in group_states]
                    )

                noise_pred_crop_batch = _predict_noise(
                    pipe=pipe,
                    latent_model_input=crop_latent_model_input,
                    timestep=timestep_batch,
                    guidance=guidance_batch,
                    prompt_embeds=prompt_embeds_batch,
                    prompt_mask=prompt_mask_batch,
                    img_shapes=img_shapes_batch,
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=true_cfg_scale,
                    negative_prompt_embeds=negative_prompt_embeds_batch,
                    negative_prompt_mask=negative_prompt_mask_batch,
                    latent_token_count=latents_batch.size(1),
                )

                start = 0
                for state, local_bs in zip(group_states, batch_sizes):
                    end = start + local_bs
                    noise_pred_crop = noise_pred_crop_batch[start:end]
                    start = end

                    crop_latents_dtype = state["latents"].dtype
                    state["latents"] = state["scheduler"].step(
                        noise_pred_crop, t, state["latents"], return_dict=False
                    )[0]
                    if state["latents"].dtype != crop_latents_dtype:
                        state["latents"] = state["latents"].to(crop_latents_dtype)

                    if state["latent_bbox"] is not None:
                        y1_l, y2_l, x1_l, x2_l = state["latent_bbox"]
                        crop_latents_img = _unpack_latents(
                            pipe,
                            state["latents"],
                            state["height"],
                            state["width"],
                        )
                        target_h = y2_l - y1_l
                        target_w = x2_l - x1_l
                        crop_latents_img = _resize_latents(
                            crop_latents_img, target_h, target_w
                        )
                        base_full_img[:, :, :, y1_l:y2_l, x1_l:x2_l] = crop_latents_img

            if any_bbox:
                full_latents = _pack_latents(pipe, base_full_img)

        else:
            full_latent_model_input = torch.cat(
                [full_latents, full_state["cond_latents"]], dim=1
            )
            noise_pred_full = _predict_noise(
                pipe=pipe,
                latent_model_input=full_latent_model_input,
                timestep=timestep,
                guidance=guidance,
                prompt_embeds=full_state["prompt_embeds"],
                prompt_mask=full_state["prompt_mask"],
                img_shapes=full_state["img_shapes"],
                do_true_cfg=do_true_cfg,
                true_cfg_scale=true_cfg_scale,
                negative_prompt_embeds=full_state["negative_prompt_embeds"],
                negative_prompt_mask=full_state["negative_prompt_mask"],
                latent_token_count=full_latents.size(1),
            )

            full_latents_dtype = full_latents.dtype
            full_latents = full_scheduler.step(
                noise_pred_full, t, full_latents, return_dict=False
            )[0]
            if full_latents.dtype != full_latents_dtype:
                full_latents = full_latents.to(full_latents_dtype)

            if any_bbox:
                full_latents_img = _unpack_latents(
                    pipe,
                    full_latents,
                    full_state["height"],
                    full_state["width"],
                )
                base = base_full_img
                for state in crop_states:
                    if state["latent_bbox"] is None:
                        continue
                    y1_l, y2_l, x1_l, x2_l = state["latent_bbox"]
                    base[:, :, :, y1_l:y2_l, x1_l:x2_l] = full_latents_img[
                        :, :, :, y1_l:y2_l, x1_l:x2_l
                    ]
                full_latents = _pack_latents(pipe, base)

    latents = _unpack_latents(
        pipe,
        full_latents,
        full_state["height"],
        full_state["width"],
    )
    latents = latents.to(pipe.vae.dtype)

    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean

    image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return image
