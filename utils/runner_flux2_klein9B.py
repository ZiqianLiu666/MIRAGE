import copy
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    compute_empirical_mu,
    retrieve_timesteps,
)

from .geometry import (
    add_noise_like,
    aligned_crop,
    bbox_to_latent_coords,
    coerce_bbox,
    make_noise_like,
    min_size_latent_bbox,
)
from .composition import RegionComposer


def _get_device(pipe):
    if hasattr(pipe, "_execution_device") and pipe._execution_device is not None:
        return pipe._execution_device
    return pipe.transformer.device


def _preprocess_image(pipe, image: PIL.Image.Image, size=None):
    pipe.image_processor.check_image_input(image)

    if size is not None:
        image_width, image_height = size
    else:
        image_width, image_height = image.size
        if image_width * image_height > 1024 * 1024:
            image = pipe.image_processor._resize_to_target_area(image, 1024 * 1024)
            image_width, image_height = image.size

        multiple_of = pipe.vae_scale_factor * 2
        image_width = (image_width // multiple_of) * multiple_of
        image_height = (image_height // multiple_of) * multiple_of

    image_tensor = pipe.image_processor.preprocess(
        image, height=image_height, width=image_width
    )
    return image_tensor, (image_width, image_height)


def _should_use_cfg(pipe, guidance_scale: float) -> bool:
    is_distilled = bool(getattr(getattr(pipe, "config", None), "is_distilled", False))
    return float(guidance_scale) > 1.0 and not is_distilled


def _predict_noise(
    pipe,
    latents: torch.Tensor,
    cond_latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    guidance_scale: float,
    use_cfg: bool,
    negative_prompt_embeds: Optional[torch.Tensor],
    negative_text_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    model_input = torch.cat([latents, cond_latents], dim=1).to(pipe.transformer.dtype)

    cond_context = (
        pipe.transformer.cache_context("cond")
        if hasattr(pipe.transformer, "cache_context")
        else nullcontext()
    )
    with cond_context:
        noise_pred = pipe.transformer(
            hidden_states=model_input,
            timestep=timestep / 1000,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

    if use_cfg:
        uncond_context = (
            pipe.transformer.cache_context("uncond")
            if hasattr(pipe.transformer, "cache_context")
            else nullcontext()
        )
        with uncond_context:
            neg_noise_pred = pipe.transformer(
                hidden_states=model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=negative_prompt_embeds,
                txt_ids=negative_text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
        noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

    return noise_pred[:, : latents.shape[1]]


@torch.no_grad()
def run_flux2_multi_branch(
    pipe,
    full_image: PIL.Image.Image,
    full_prompt: str,
    crop_prompts: List[str],
    bboxes: List[Union[Dict[str, int], Tuple[int, int, int, int], List[int]]],
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    generator: Optional[torch.Generator] = None,
    patch_ratio: float = 0.8,
):
    device = _get_device(pipe)
    latent_dtype = pipe.transformer.dtype

    source_size = full_image.size
    full_tensor, full_size = _preprocess_image(pipe, full_image)
    full_tensor = full_tensor.to(device=device, dtype=pipe.vae.dtype)
    full_image_latents = pipe._encode_vae_image(full_tensor, generator).to(
        device=device, dtype=latent_dtype
    )
    full_latent_hw = (full_image_latents.shape[-2], full_image_latents.shape[-1])

    noise_full = make_noise_like(full_image_latents, generator)

    num_channels_latents = pipe.transformer.config.in_channels // 4
    full_latents, full_latent_ids = pipe.prepare_latents(
        batch_size=full_image_latents.shape[0],
        num_latents_channels=num_channels_latents,
        height=full_tensor.shape[-2],
        width=full_tensor.shape[-1],
        dtype=latent_dtype,
        device=device,
        generator=generator,
        latents=noise_full,
    )
    full_latent_ids = full_latent_ids.to(device)

    full_cond_latents, full_cond_ids = pipe.prepare_image_latents(
        images=[full_tensor],
        batch_size=full_latents.shape[0],
        generator=generator,
        device=device,
        dtype=pipe.vae.dtype,
    )
    full_cond_latents = full_cond_latents.to(device=device, dtype=latent_dtype)
    full_latent_image_ids = torch.cat([full_latent_ids, full_cond_ids.to(device)], dim=1)

    full_prompt_embeds, full_text_ids = pipe.encode_prompt(full_prompt, device=device)
    full_prompt_embeds = full_prompt_embeds.to(dtype=latent_dtype)

    use_cfg = _should_use_cfg(pipe, guidance_scale)
    neg_prompt_embeds = None
    neg_text_ids = None
    if use_cfg:
        neg_prompt_embeds, neg_text_ids = pipe.encode_prompt("", device=device)
        neg_prompt_embeds = neg_prompt_embeds.to(dtype=latent_dtype)

    crop_states = []
    for branch_id, (prompt, bbox) in enumerate(zip(crop_prompts, bboxes)):
        latent_bbox = min_size_latent_bbox(
            bbox_to_latent_coords(coerce_bbox(bbox), source_size, full_latent_hw),
            full_size,
            full_latent_hw,
        )
        y1_l, y2_l, x1_l, x2_l = latent_bbox

        crop_image, crop_size = aligned_crop(
            full_image, full_size, full_latent_hw, latent_bbox
        )
        crop_tensor, _ = _preprocess_image(pipe, crop_image, size=crop_size)
        crop_tensor = crop_tensor.to(device=device, dtype=pipe.vae.dtype)

        window = noise_full[..., y1_l:y2_l, x1_l:x2_l].contiguous()
        crop_latents, crop_latent_ids = pipe.prepare_latents(
            batch_size=full_latents.shape[0],
            num_latents_channels=num_channels_latents,
            height=crop_tensor.shape[-2],
            width=crop_tensor.shape[-1],
            dtype=latent_dtype,
            device=device,
            generator=generator,
            latents=window,
        )
        crop_latent_ids = crop_latent_ids.to(device)

        crop_cond_latents, crop_cond_ids = pipe.prepare_image_latents(
            images=[crop_tensor],
            batch_size=full_latents.shape[0],
            generator=generator,
            device=device,
            dtype=pipe.vae.dtype,
        )
        prompt_embeds, text_ids = pipe.encode_prompt(prompt, device=device)

        crop_states.append(
            {
                "latents": crop_latents,
                "latent_ids": crop_latent_ids,
                "cond_latents": crop_cond_latents.to(device=device, dtype=latent_dtype),
                "latent_image_ids": torch.cat(
                    [crop_latent_ids, crop_cond_ids.to(device)], dim=1
                ),
                "prompt_embeds": prompt_embeds.to(dtype=latent_dtype),
                "text_ids": text_ids,
                "scheduler": copy.deepcopy(pipe.scheduler),
                "image_latents": pipe._encode_vae_image(crop_tensor, generator).to(
                    device=device, dtype=latent_dtype
                ),
                "noise": window,
                "latent_bbox": latent_bbox,
                "branch_id": branch_id,
            }
        )

    full_scheduler = copy.deepcopy(pipe.scheduler)

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if (
        hasattr(full_scheduler.config, "use_flow_sigmas")
        and full_scheduler.config.use_flow_sigmas
    ):
        sigmas = None

    mu = compute_empirical_mu(
        image_seq_len=full_latent_hw[0] * full_latent_hw[1],
        num_steps=num_inference_steps,
    )

    timesteps, num_inference_steps = retrieve_timesteps(
        full_scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
    )
    for state in crop_states:
        retrieve_timesteps(
            state["scheduler"], num_inference_steps, device, sigmas=sigmas, mu=mu
        )

    num_steps = len(timesteps)
    patch_until = int(num_steps * max(0.0, min(float(patch_ratio), 1.0)))

    def reference_at(next_timestep):
        if next_timestep is None:
            return full_image_latents.clone()
        return add_noise_like(
            full_scheduler, full_image_latents, noise_full, next_timestep
        )

    handover_index = patch_until - 1

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
                noise_pred = _predict_noise(
                    pipe=pipe,
                    latents=state["latents"],
                    cond_latents=state["cond_latents"],
                    timestep=timestep,
                    prompt_embeds=state["prompt_embeds"],
                    text_ids=state["text_ids"],
                    latent_image_ids=state["latent_image_ids"],
                    guidance_scale=guidance_scale,
                    use_cfg=use_cfg,
                    negative_prompt_embeds=neg_prompt_embeds,
                    negative_text_ids=neg_text_ids,
                )
                state["latents"] = state["scheduler"].step(
                    noise_pred, t, state["latents"], return_dict=False
                )[0].to(latent_dtype)

                if composer is None:
                    continue

                branch = pipe._unpack_latents_with_ids(
                    state["latents"], state["latent_ids"]
                )
                reference = (
                    state["image_latents"]
                    if next_t is None
                    else add_noise_like(
                        state["scheduler"],
                        state["image_latents"],
                        state["noise"],
                        next_t,
                    )
                )
                composer.add(
                    branch, state["latent_bbox"], branch - reference, state["branch_id"]
                )

            if composer is not None:
                full_latents = pipe._pack_latents(composer.compose())

        else:
            noise_pred = _predict_noise(
                pipe=pipe,
                latents=full_latents,
                cond_latents=full_cond_latents,
                timestep=timestep,
                prompt_embeds=full_prompt_embeds,
                text_ids=full_text_ids,
                latent_image_ids=full_latent_image_ids,
                guidance_scale=guidance_scale,
                use_cfg=use_cfg,
                negative_prompt_embeds=neg_prompt_embeds,
                negative_text_ids=neg_text_ids,
            )
            full_latents = full_scheduler.step(
                noise_pred, t, full_latents, return_dict=False
            )[0].to(latent_dtype)

            canvas = reference_at(next_t)
            denoised = pipe._unpack_latents_with_ids(full_latents, full_latent_ids)
            for state in crop_states:
                y1_l, y2_l, x1_l, x2_l = state["latent_bbox"]
                canvas[:, :, y1_l:y2_l, x1_l:x2_l] = denoised[
                    :, :, y1_l:y2_l, x1_l:x2_l
                ]
            full_latents = pipe._pack_latents(canvas)

    latents = pipe._unpack_latents_with_ids(full_latents, full_latent_ids)
    latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents_bn_std = torch.sqrt(
        pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    latents = pipe._unpatchify_latents(latents * latents_bn_std + latents_bn_mean)

    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]
