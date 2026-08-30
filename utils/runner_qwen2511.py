import copy
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)

from .geometry import (
    add_noise_like,
    aligned_crop,
    bbox_to_latent_coords,
    coerce_bbox,
    min_size_latent_bbox,
    region_write_weight,
)
from .composition import RegionComposer


WRITE_MARGIN_CELLS = 2

BOX_WRITE_WEIGHT = 1.0

BRANCH_MAX_DOWNSCALE = 4.0

BRANCH_CONTEXT_CELLS = 2

MATCH_BASELINE_CANVAS = False


def _get_device(pipe):
    if hasattr(pipe, "_execution_device") and pipe._execution_device is not None:
        return pipe._execution_device
    if hasattr(pipe, "device"):
        return pipe.device
    return pipe.transformer.device


def _token_grid_hw(pipe, height: int, width: int) -> Tuple[int, int]:
    multiple_of = pipe.vae_scale_factor * 2
    return height // multiple_of, width // multiple_of


def _tokens_to_grid(tokens: torch.Tensor, latent_hw: Tuple[int, int]) -> torch.Tensor:
    latent_h, latent_w = latent_hw
    batch, token_count, channels = tokens.shape
    return tokens.reshape(batch, latent_h, latent_w, channels).permute(0, 3, 1, 2)


def _grid_to_tokens(grid: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = grid.shape
    return grid.permute(0, 2, 3, 1).reshape(batch, height * width, channels)


def _prepare_state(
    pipe,
    image: PIL.Image.Image,
    prompt: str,
    negative_prompt: Optional[Union[str, List[str]]],
    do_true_cfg: bool,
    device: torch.device,
    latent_dtype: torch.dtype,
    target_area: float = VAE_IMAGE_SIZE,
):
    image = image.convert("RGB")
    image_w, image_h = image.size

    multiple_of = pipe.vae_scale_factor * 2
    if MATCH_BASELINE_CANVAS and image_w * image_h <= target_area:
        width, height = image_w, image_h
    else:
        width, height = calculate_dimensions(target_area, image_w / image_h)
    width = int(width) // multiple_of * multiple_of
    height = int(height) // multiple_of * multiple_of

    condition_w, condition_h = calculate_dimensions(
        target_area * (CONDITION_IMAGE_SIZE / VAE_IMAGE_SIZE), image_w / image_h
    )
    condition_image = pipe.image_processor.resize(
        image, int(condition_h), int(condition_w)
    )

    vae_tensor = pipe.image_processor.preprocess(image, height=height, width=width)
    if vae_tensor.ndim == 3:
        vae_tensor = vae_tensor.unsqueeze(0)
    vae_tensor = vae_tensor.unsqueeze(2).to(device=device, dtype=pipe.vae.dtype)

    prompt_embeds, prompt_mask = pipe.encode_prompt(
        image=[condition_image], prompt=prompt, device=device, num_images_per_prompt=1
    )

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

    latent_h, latent_w = _token_grid_hw(pipe, height, width)

    return {
        "input_size": (image_w, image_h),
        "height": height,
        "width": width,
        "latent_hw": (latent_h, latent_w),
        "vae_tensor": vae_tensor,
        "prompt_embeds": prompt_embeds.to(dtype=latent_dtype),
        "prompt_mask": prompt_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_mask": negative_prompt_mask,
        "img_shapes": [[(1, latent_h, latent_w), (1, latent_h, latent_w)]],
    }


def _forward(
    pipe,
    model_input: torch.Tensor,
    token_count: int,
    timestep: torch.Tensor,
    guidance: Optional[torch.Tensor],
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    img_shapes,
    cache_name: str,
):
    context = (
        pipe.transformer.cache_context(cache_name)
        if hasattr(pipe.transformer, "cache_context")
        else nullcontext()
    )
    with context:
        return pipe.transformer(
            hidden_states=model_input,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            return_dict=False,
        )[0][:, :token_count]


def _predict_noise(
    pipe,
    latents: torch.Tensor,
    cond_latents: torch.Tensor,
    timestep: torch.Tensor,
    guidance: Optional[torch.Tensor],
    state: Dict,
    do_true_cfg: bool,
    true_cfg_scale: float,
):
    model_input = torch.cat([latents, cond_latents], dim=1)
    token_count = latents.shape[1]

    noise_pred = _forward(
        pipe,
        model_input,
        token_count,
        timestep,
        guidance,
        state["prompt_embeds"],
        state["prompt_mask"],
        state["img_shapes"],
        "cond",
    )
    if not do_true_cfg:
        return noise_pred

    neg_noise_pred = _forward(
        pipe,
        model_input,
        token_count,
        timestep,
        guidance,
        state["negative_prompt_embeds"],
        state["negative_prompt_mask"],
        state["img_shapes"],
        "uncond",
    )

    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    return comb_pred * (cond_norm / noise_norm)


def _per_step_write_weight(effective: torch.Tensor, steps: int) -> torch.Tensor:
    target = effective.to(torch.float32)
    low = torch.zeros_like(target)
    high = torch.ones_like(target)
    for _ in range(60):
        mid = 0.5 * (low + high)
        survived = mid * (1.0 - mid.pow(steps)) / (
            steps * (1.0 - mid).clamp(min=1e-12)
        )
        below = survived < target
        low = torch.where(below, mid, low)
        high = torch.where(below, high, mid)
    w = 0.5 * (low + high)
    w = torch.where(target >= 1.0, torch.ones_like(w), w)
    w = torch.where(target <= 0.0, torch.zeros_like(w), w)
    return w.to(effective.dtype)


@torch.no_grad()
def run_qwen_multi_branch(
    pipe,
    full_image: PIL.Image.Image,
    full_prompt: str,
    crop_prompts: List[str],
    bboxes: List[Union[Dict[str, int], Tuple[int, int, int, int], List[int]]],
    num_inference_steps: int = 40,
    true_cfg_scale: float = 4.0,
    guidance_scale: Optional[float] = 1.0,
    negative_prompt: Optional[Union[str, List[str]]] = " ",
    generator: Optional[torch.Generator] = None,
    patch_ratio: float = 0.8,
):

    device = _get_device(pipe)
    latent_dtype = pipe.transformer.dtype
    do_true_cfg = true_cfg_scale > 1.0 and negative_prompt is not None

    guidance = None
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)

    full_state = _prepare_state(
        pipe=pipe,
        image=full_image,
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        do_true_cfg=do_true_cfg,
        device=device,
        latent_dtype=latent_dtype,
    )
    full_latent_hw = full_state["latent_hw"]
    multiple_of = pipe.vae_scale_factor * 2

    num_channels_latents = pipe.transformer.config.in_channels // 4
    noise_full, full_cond_latents = pipe.prepare_latents(
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
    noise_full = noise_full.to(device=device, dtype=latent_dtype)
    full_state["cond_latents"] = full_cond_latents.to(device=device, dtype=latent_dtype)

    noise_full_grid = _tokens_to_grid(noise_full, full_latent_hw)
    full_image_grid = _tokens_to_grid(full_state["cond_latents"], full_latent_hw)
    full_latents = noise_full

    crop_states = []
    for branch_id, (prompt, bbox) in enumerate(zip(crop_prompts, bboxes)):
        latent_bbox = min_size_latent_bbox(
            bbox_to_latent_coords(
                coerce_bbox(bbox), full_state["input_size"], full_latent_hw
            ),
            (full_state["width"], full_state["height"]),
            full_latent_hw,
        )
        y1_b, y2_b, x1_b, x2_b = latent_bbox
        crop_bbox = (
            max(0, y1_b - BRANCH_CONTEXT_CELLS),
            min(full_latent_hw[0], y2_b + BRANCH_CONTEXT_CELLS),
            max(0, x1_b - BRANCH_CONTEXT_CELLS),
            min(full_latent_hw[1], x2_b + BRANCH_CONTEXT_CELLS),
        )
        state_offset = (y1_b - crop_bbox[0], x1_b - crop_bbox[2])

        crop_image, _ = aligned_crop(
            full_image,
            (full_state["width"], full_state["height"]),
            full_latent_hw,
            crop_bbox,
        )

        box_pixels = (
            (crop_bbox[1] - crop_bbox[0])
            * (crop_bbox[3] - crop_bbox[2])
            * (multiple_of * multiple_of)
        )
        target_area = min(
            VAE_IMAGE_SIZE,
            max(
                box_pixels * BRANCH_MAX_DOWNSCALE * BRANCH_MAX_DOWNSCALE,
                pipe.scheduler.config.get("base_image_seq_len", 256)
                * multiple_of
                * multiple_of,
            ),
        )
        state = _prepare_state(
            pipe=pipe,
            image=crop_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_true_cfg=do_true_cfg,
            device=device,
            latent_dtype=latent_dtype,
            target_area=target_area,
        )
        branch_latents, branch_cond_latents = pipe.prepare_latents(
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
        state["latents"] = branch_latents.to(device=device, dtype=latent_dtype)
        state["cond_latents"] = branch_cond_latents.to(
            device=device, dtype=latent_dtype
        )
        state["latent_bbox"] = latent_bbox
        state["crop_bbox"] = crop_bbox
        state["state_offset"] = state_offset
        state["branch_id"] = branch_id
        state["scheduler"] = copy.deepcopy(pipe.scheduler)
        crop_states.append(state)

    full_scheduler = copy.deepcopy(pipe.scheduler)
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

    def shift_for(token_count: int) -> float:
        return calculate_shift(
            token_count,
            full_scheduler.config.get("base_image_seq_len", 256),
            full_scheduler.config.get("max_image_seq_len", 4096),
            full_scheduler.config.get("base_shift", 0.5),
            full_scheduler.config.get("max_shift", 1.15),
        )

    timesteps, num_inference_steps = retrieve_timesteps(
        full_scheduler, num_inference_steps, device, sigmas=sigmas, mu=shift_for(
            full_latents.shape[1]
        )
    )
    for state in crop_states:
        retrieve_timesteps(
            state["scheduler"],
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=shift_for(state["latents"].shape[1]),
        )

    num_steps = len(timesteps)
    patch_until = int(num_steps * max(0.0, min(float(patch_ratio), 1.0)))
    handover_index = patch_until - 1

    if guidance is not None:
        guidance = guidance.expand(full_latents.shape[0])

    def reference_at(next_timestep):
        if next_timestep is None:
            return full_image_grid.clone()
        return add_noise_like(
            full_scheduler, full_image_grid, noise_full_grid, next_timestep
        )

    write_weight = _per_step_write_weight(
        region_write_weight(
            full_latent_hw,
            [state["latent_bbox"] for state in crop_states],
            WRITE_MARGIN_CELLS,
            device=full_image_grid.device,
            dtype=full_image_grid.dtype,
        ).clamp(max=BOX_WRITE_WEIGHT),
        max(num_steps - patch_until, 1),
    )

    if handover_index >= 0:
        print(
            f"[handover] step {patch_until}/{num_steps}, "
            f"sigma={float(full_scheduler.sigmas[patch_until]):.3f}, "
            f"branch tokens {[st['latents'].shape[1] for st in crop_states]} "
            f"vs canvas {full_latents.shape[1]}"
        )
    print(
        f"[write region] boxes cover "
        f"{float((write_weight >= BOX_WRITE_WEIGHT).float().mean()):.1%} of the "
        f"canvas, collar adds "
        f"{float(((write_weight > 0) & (write_weight < BOX_WRITE_WEIGHT)).float().mean()):.1%} "
        f"at partial weight"
    )

    for step_index, t in enumerate(
        tqdm(timesteps, desc="Diffusion steps", leave=False)
    ):
        next_t = timesteps[step_index + 1] if step_index + 1 < num_steps else None
        timestep = t.expand(full_latents.shape[0]).to(full_latents.dtype)

        if step_index < patch_until:
            composer = (
                RegionComposer(reference_at(next_t))
                if step_index == handover_index
                else None
            )

            for state in crop_states:
                branch_timestep = t.expand(state["latents"].shape[0]).to(
                    state["latents"].dtype
                )
                noise_pred = _predict_noise(
                    pipe=pipe,
                    latents=state["latents"],
                    cond_latents=state["cond_latents"],
                    timestep=branch_timestep,
                    guidance=guidance,
                    state=state,
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=true_cfg_scale,
                )

                if composer is None:
                    state["latents"] = state["scheduler"].step(
                        noise_pred, t, state["latents"], return_dict=False
                    )[0].to(latent_dtype)
                    continue

                sigma = state["scheduler"].sigmas[step_index].to(latent_dtype)
                clean = state["latents"] - sigma * noise_pred

                y1_l, y2_l, x1_l, x2_l = state["latent_bbox"]
                cy1, cy2, cx1, cx2 = state["crop_bbox"]
                oy, ox = state["state_offset"]
                clean_crop = torch.nn.functional.adaptive_avg_pool2d(
                    _tokens_to_grid(clean, state["latent_hw"]).float(),
                    (cy2 - cy1, cx2 - cx1),
                ).to(latent_dtype)
                clean_box = clean_crop[
                    ..., oy : oy + (y2_l - y1_l), ox : ox + (x2_l - x1_l)
                ].contiguous()

                window_reference = full_image_grid[..., y1_l:y2_l, x1_l:x2_l]
                branch_z = (
                    clean_box
                    if next_t is None
                    else add_noise_like(
                        full_scheduler,
                        clean_box,
                        noise_full_grid[..., y1_l:y2_l, x1_l:x2_l],
                        next_t,
                    )
                )
                composer.add(
                    branch_z,
                    state["latent_bbox"],
                    clean_box - window_reference,
                    state["branch_id"],
                )

            if composer is not None:
                full_latents = _grid_to_tokens(composer.compose())

        else:
            noise_pred = _predict_noise(
                pipe=pipe,
                latents=full_latents,
                cond_latents=full_state["cond_latents"],
                timestep=timestep,
                guidance=guidance,
                state=full_state,
                do_true_cfg=do_true_cfg,
                true_cfg_scale=true_cfg_scale,
            )
            full_latents = full_scheduler.step(
                noise_pred, t, full_latents, return_dict=False
            )[0].to(latent_dtype)

            canvas = reference_at(next_t)
            denoised = _tokens_to_grid(full_latents, full_latent_hw)
            canvas = canvas + write_weight * (denoised.to(canvas.dtype) - canvas)
            full_latents = _grid_to_tokens(canvas)

    latents = pipe._unpack_latents(
        full_latents, full_state["height"], full_state["width"], pipe.vae_scale_factor
    ).to(pipe.vae.dtype)

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
    return pipe.image_processor.postprocess(image, output_type="pil")[0]
