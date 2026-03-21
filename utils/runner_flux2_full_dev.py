import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from diffusers.pipelines.flux2.pipeline_flux2 import (
    compute_empirical_mu,
    retrieve_timesteps,
)

from .geometry import bbox_to_latent_coords, coerce_bbox

from diffusers.utils.torch_utils import randn_tensor

def _get_device(pipe):
    if hasattr(pipe, "_execution_device") and pipe._execution_device is not None:
        return pipe._execution_device
    return pipe.transformer.device


def _preprocess_image(pipe, image: PIL.Image.Image):
    pipe.image_processor.check_image_input(image)

    image_width, image_height = image.size
    if image_width * image_height > 1024 * 1024:
        image = pipe.image_processor._resize_to_target_area(image, 1024 * 1024)
        image_width, image_height = image.size

    multiple_of = pipe.vae_scale_factor * 2
    image_width = (image_width // multiple_of) * multiple_of
    image_height = (image_height // multiple_of) * multiple_of

    image_tensor = pipe.image_processor.preprocess(
        image, height=image_height, width=image_width, resize_mode="crop"
    )
    return image_tensor, (image_width, image_height)


def _make_noise_like(latents: torch.Tensor, generator: Optional[torch.Generator]):
    if generator is None:
        return torch.randn_like(latents)
    return randn_tensor(
        latents.shape, generator=generator, device=latents.device, dtype=latents.dtype
    )


def _resize_latents(latents: torch.Tensor, target_h: int, target_w: int):
    _, _, h, w = latents.shape
    if h == target_h and w == target_w:
        return latents
    return F.interpolate(
        latents, size=(target_h, target_w), mode="bilinear", align_corners=False
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


def _group_crop_state_indices(crop_states: List[Dict[str, Any]]):
    grouped = defaultdict(list)
    for idx, state in enumerate(crop_states):
        key = (
            state["latents"].shape[1],
            state["cond_latents"].shape[1],
            tuple(state["latent_image_ids"].shape),
            tuple(state["prompt_embeds"].shape),
            tuple(state["text_ids"].shape),
        )
        grouped[key].append(idx)
    return list(grouped.values())


def _merge_id_tensors(id_tensors: List[torch.Tensor]):
    first = id_tensors[0]
    if first.ndim >= 2 and first.shape[0] == 1:
        return torch.cat(id_tensors, dim=0)
    return first


@torch.no_grad()
def run_flux2_multi_branch(
    pipe,
    full_image: PIL.Image.Image,
    crop_images: List[PIL.Image.Image],
    full_prompt: str,
    crop_prompts: List[str],
    bboxes: List[Optional[Union[Dict[str, int], Tuple[int, int, int, int], List[int]]]],
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    generator: Optional[torch.Generator] = None,
    patch_ratio: float = 0.8,
):
    device = _get_device(pipe)
    full_tensor, full_size = _preprocess_image(pipe, full_image)
    full_tensor = full_tensor.to(device=device, dtype=pipe.vae.dtype)

    crop_tensors = []
    for img in crop_images:
        crop_tensor, _ = _preprocess_image(pipe, img)
        crop_tensors.append(crop_tensor.to(device=device, dtype=pipe.vae.dtype))

    full_prompt_embeds, full_text_ids = pipe.encode_prompt(full_prompt, device=device)
    full_prompt_embeds = full_prompt_embeds.to(dtype=pipe.transformer.dtype)
    latent_dtype = full_prompt_embeds.dtype

    crop_prompt_embeds_list = []
    crop_text_ids_list = []
    for prompt in crop_prompts:
        prompt_embeds, text_ids = pipe.encode_prompt(prompt, device=device)
        crop_prompt_embeds_list.append(prompt_embeds.to(dtype=pipe.transformer.dtype))
        crop_text_ids_list.append(text_ids)

    full_image_latents = pipe._encode_vae_image(full_tensor, generator).to(
        device=device, dtype=latent_dtype
    )
    crop_image_latents_list = [
        pipe._encode_vae_image(t, generator).to(device=device, dtype=latent_dtype)
        for t in crop_tensors
    ]

    noise_full = _make_noise_like(full_image_latents, generator)
    noise_crops = [
        _make_noise_like(crop_latents, generator)
        for crop_latents in crop_image_latents_list
    ]

    full_scheduler = copy.deepcopy(pipe.scheduler)
    crop_schedulers = [copy.deepcopy(pipe.scheduler) for _ in crop_images]

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if (
        hasattr(full_scheduler.config, "use_flow_sigmas")
        and full_scheduler.config.use_flow_sigmas
    ):
        sigmas = None

    image_seq_len = full_image_latents.shape[-2] * full_image_latents.shape[-1]
    mu = compute_empirical_mu(
        image_seq_len=image_seq_len, num_steps=num_inference_steps
    )

    timesteps, num_inference_steps = retrieve_timesteps(
        full_scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
    )
    for sch in crop_schedulers:
        retrieve_timesteps(sch, num_inference_steps, device, sigmas=sigmas, mu=mu)

    num_steps = len(timesteps)
    patch_ratio = float(patch_ratio)
    patch_ratio = max(0.0, min(patch_ratio, 1.0))
    patch_until = int(num_steps * patch_ratio)

    num_channels_latents = pipe.transformer.config.in_channels // 4
    full_latents, full_latent_ids = pipe.prepare_latents(
        batch_size=full_image_latents.shape[0],
        num_latents_channels=num_channels_latents,
        height=full_tensor.shape[-2],
        width=full_tensor.shape[-1],
        dtype=latent_dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    full_latents = full_latents.to(device=device, dtype=latent_dtype)
    full_latent_ids = full_latent_ids.to(device)

    crop_latents_list = []
    crop_latent_ids_list = []
    for crop_tensor in crop_tensors:
        crop_latents, crop_latent_ids = pipe.prepare_latents(
            batch_size=full_image_latents.shape[0],
            num_latents_channels=num_channels_latents,
            height=crop_tensor.shape[-2],
            width=crop_tensor.shape[-1],
            dtype=latent_dtype,
            device=device,
            generator=generator,
            latents=None,
        )
        crop_latents_list.append(crop_latents.to(device=device, dtype=latent_dtype))
        crop_latent_ids_list.append(crop_latent_ids.to(device))

    batch_size = full_latents.shape[0]
    full_cond_latents, full_cond_ids = pipe.prepare_image_latents(
        images=[full_tensor],
        batch_size=batch_size,
        generator=generator,
        device=device,
        dtype=pipe.vae.dtype,
    )
    full_cond_latents = full_cond_latents.to(device=device, dtype=latent_dtype)
    full_cond_ids = full_cond_ids.to(device)
    full_latent_image_ids = torch.cat([full_latent_ids, full_cond_ids], dim=1)

    crop_states = []
    for i, crop_latents in enumerate(crop_latents_list):
        crop_cond_latents, crop_cond_ids = pipe.prepare_image_latents(
            images=[crop_tensors[i]],
            batch_size=batch_size,
            generator=generator,
            device=device,
            dtype=pipe.vae.dtype,
        )
        crop_cond_latents = crop_cond_latents.to(device=device, dtype=latent_dtype)
        crop_cond_ids = crop_cond_ids.to(device)
        crop_latent_image_ids = torch.cat(
            [crop_latent_ids_list[i], crop_cond_ids], dim=1
        )

        latent_bbox = None
        if bboxes[i] is not None:
            latent_bbox = bbox_to_latent_coords(
                coerce_bbox(bboxes[i]),
                full_size,
                (
                    full_image_latents.shape[-2],
                    full_image_latents.shape[-1],
                ),
            )

        crop_states.append(
            {
                "latents": crop_latents,
                "latent_ids": crop_latent_ids_list[i],
                "cond_latents": crop_cond_latents,
                "latent_image_ids": crop_latent_image_ids,
                "prompt_embeds": crop_prompt_embeds_list[i],
                "text_ids": crop_text_ids_list[i],
                "scheduler": crop_schedulers[i],
                "noise": noise_crops[i],
                "latent_bbox": latent_bbox,
            }
        )

    any_bbox = any(state["latent_bbox"] is not None for state in crop_states)
    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
    guidance = guidance.expand(full_latents.shape[0])

    for step_index, t in enumerate(
        tqdm(timesteps, desc="Diffusion steps", leave=False)
    ):
        base_full = None
        if any_bbox:
            base_full = _add_noise_like(
                full_scheduler, full_image_latents, noise_full, t
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
                prompt_embeds_batch = torch.cat(
                    [state["prompt_embeds"] for state in group_states], dim=0
                )
                text_ids_batch = _merge_id_tensors(
                    [state["text_ids"] for state in group_states]
                )
                latent_image_ids_batch = _merge_id_tensors(
                    [state["latent_image_ids"] for state in group_states]
                )
                guidance_batch = guidance.expand(latents_batch.shape[0])
                timestep_batch = t.expand(latents_batch.shape[0]).to(
                    latents_batch.dtype
                )

                noise_pred_crop_batch = pipe.transformer(
                    hidden_states=crop_latent_model_input.to(pipe.transformer.dtype),
                    timestep=timestep_batch / 1000,
                    guidance=guidance_batch,
                    encoder_hidden_states=prompt_embeds_batch,
                    txt_ids=text_ids_batch,
                    img_ids=latent_image_ids_batch,
                    return_dict=False,
                )[0]
                noise_pred_crop_batch = noise_pred_crop_batch[:, : latents_batch.size(1)]

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
                        crop_latents_img = pipe._unpack_latents_with_ids(
                            state["latents"], state["latent_ids"]
                        )
                        target_h = y2_l - y1_l
                        target_w = x2_l - x1_l
                        crop_latents_img = _resize_latents(
                            crop_latents_img, target_h, target_w
                        )

                        base_full[:, :, y1_l:y2_l, x1_l:x2_l] = crop_latents_img

            if any_bbox:
                full_latents = pipe._pack_latents(base_full)

        else:
            full_latent_model_input = torch.cat(
                [full_latents, full_cond_latents], dim=1
            )
            noise_pred_full = pipe.transformer(
                hidden_states=full_latent_model_input.to(pipe.transformer.dtype),
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=full_prompt_embeds,
                txt_ids=full_text_ids,
                img_ids=full_latent_image_ids,
                return_dict=False,
            )[0]
            noise_pred_full = noise_pred_full[:, : full_latents.size(1)]

            full_latents_dtype = full_latents.dtype
            full_latents = full_scheduler.step(
                noise_pred_full, t, full_latents, return_dict=False
            )[0]
            if full_latents.dtype != full_latents_dtype:
                full_latents = full_latents.to(full_latents_dtype)

            if any_bbox:
                full_latents_img = pipe._unpack_latents_with_ids(
                    full_latents, full_latent_ids
                )
                base = base_full
                for state in crop_states:
                    if state["latent_bbox"] is None:
                        continue
                    y1_l, y2_l, x1_l, x2_l = state["latent_bbox"]
                    base[:, :, y1_l:y2_l, x1_l:x2_l] = full_latents_img[
                        :, :, y1_l:y2_l, x1_l:x2_l
                    ]

                full_latents = pipe._pack_latents(base)

    latents = pipe._unpack_latents_with_ids(full_latents, full_latent_ids)
    latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents_bn_std = torch.sqrt(
        pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    latents = latents * latents_bn_std + latents_bn_mean
    latents = pipe._unpatchify_latents(latents)

    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return image
