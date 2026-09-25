import copy

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from .composition import compose
from .geometry import bbox_to_latent, crop_cells, min_size_box


def _preprocess(pipe, image):
    width, height = image.size
    if width * height > 1024 * 1024:
        image = pipe.image_processor._resize_to_target_area(image, 1024 * 1024)
        width, height = image.size
    multiple_of = pipe.vae_scale_factor * 2
    width, height = width // multiple_of * multiple_of, height // multiple_of * multiple_of
    return pipe.image_processor.preprocess(image, height=height, width=width)


def _forward(pipe, latents, branch, text, timestep, guidance):
    hidden_states = torch.cat([latents, branch["cond"]], dim=1).to(pipe.transformer.dtype)
    return pipe.transformer(
        hidden_states=hidden_states,
        timestep=timestep / 1000,
        guidance=guidance,
        encoder_hidden_states=text[0],
        txt_ids=text[1],
        img_ids=branch["img_ids"],
        return_dict=False,
    )[0][:, : latents.shape[1]]


@torch.no_grad()
def run_flux2(
    pipe,
    image,
    prompt,
    sub_prompts,
    bboxes,
    num_inference_steps=50,
    guidance_scale=4.0,
    patch_ratio=0.4,
    generator=None,
):
    """MIRAGE on FLUX.2 [dev] or [klein]; `bboxes` are pixel boxes of `image`."""
    device = pipe._execution_device
    dtype = pipe.transformer.dtype
    channels = pipe.transformer.config.in_channels // 4

    # [dev] is guidance-distilled, [klein] base uses classifier-free guidance.
    if isinstance(pipe, Flux2KleinPipeline):
        guidance = None
        use_cfg = guidance_scale > 1 and not pipe.config.is_distilled
    else:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        use_cfg = False

    def encode_text(text):
        embeds, text_ids = pipe.encode_prompt(text, device=device)
        return embeds.to(dtype), text_ids

    negative = encode_text("") if use_cfg else None

    def encode(img, text, noise=None):
        tensor = _preprocess(pipe, img).to(device=device, dtype=pipe.vae.dtype)
        cond, cond_ids = pipe.prepare_image_latents([tensor], 1, generator, device, pipe.vae.dtype)
        cond = cond.to(dtype)
        clean = pipe._unpack_latents_with_ids(cond, cond_ids)
        if noise is None:
            noise = randn_tensor(clean.shape, generator=generator, device=device, dtype=dtype)
        latents, ids = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=channels,
            height=tensor.shape[-2],
            width=tensor.shape[-1],
            dtype=dtype,
            device=device,
            generator=generator,
            latents=noise,
        )
        return {
            "size": (tensor.shape[-1], tensor.shape[-2]),
            "latents": latents,
            "ids": ids,
            "cond": cond,
            "img_ids": torch.cat([ids, cond_ids], dim=1),
            "clean": clean,
            "noise": noise,
            "text": encode_text(text),
        }

    def velocity(latents, branch, timestep):
        v = _forward(pipe, latents, branch, branch["text"], timestep, guidance)
        if negative is None:
            return v
        v_neg = _forward(pipe, latents, branch, negative, timestep, guidance)
        return v_neg + guidance_scale * (v - v_neg)

    full = encode(image, prompt)
    latent_hw = full["clean"].shape[-2:]

    # Each region branch denoises its own crop, starting from the matching window of the
    # global noise so that its latents can be pasted back into the global canvas.
    branches = []
    for sub_prompt, bbox in zip(sub_prompts, bboxes):
        box = min_size_box(bbox_to_latent(bbox, image.size, latent_hw), full["size"], latent_hw)
        y1, y2, x1, x2 = box
        crop = crop_cells(image, full["size"], latent_hw, box)
        branch = encode(crop, sub_prompt, full["noise"][..., y1:y2, x1:x2].contiguous())
        branch["box"] = box
        branch["scheduler"] = copy.deepcopy(pipe.scheduler)
        branches.append(branch)

    scheduler = copy.deepcopy(pipe.scheduler)
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = compute_empirical_mu(image_seq_len=latent_hw[0] * latent_hw[1], num_steps=num_inference_steps)
    timesteps, _ = retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
    for branch in branches:
        retrieve_timesteps(branch["scheduler"], num_inference_steps, device, sigmas=sigmas, mu=mu)

    def noised(clean, noise, i):
        # forward diffusion of the reference to the noise level reached after step i
        sigma = scheduler.sigmas[i + 1].to(clean.dtype)
        return sigma * noise + (1 - sigma) * clean

    handover = int(len(timesteps) * patch_ratio)
    latents = full["latents"]
    for i, t in enumerate(tqdm(timesteps, leave=False)):
        timestep = t.expand(1).to(dtype)
        if i < handover:
            for branch in branches:
                v = velocity(branch["latents"], branch, timestep)
                branch["latents"] = branch["scheduler"].step(v, t, branch["latents"], return_dict=False)[0]
            if i == handover - 1:
                pasted = []
                for branch in branches:
                    z = pipe._unpack_latents_with_ids(branch["latents"], branch["ids"])
                    pasted.append((z, branch["box"], z - noised(branch["clean"], branch["noise"], i)))
                latents = pipe._pack_latents(compose(noised(full["clean"], full["noise"], i), pasted))
        else:
            v = velocity(latents, full, timestep)
            latents = scheduler.step(v, t, latents, return_dict=False)[0]
            canvas = noised(full["clean"], full["noise"], i)
            denoised = pipe._unpack_latents_with_ids(latents, full["ids"])
            for y1, y2, x1, x2 in (branch["box"] for branch in branches):
                canvas[..., y1:y2, x1:x2] = denoised[..., y1:y2, x1:x2]
            latents = pipe._pack_latents(canvas)

    latents = pipe._unpack_latents_with_ids(latents, full["ids"])
    bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps)
    latents = pipe._unpatchify_latents(latents * bn_std.to(latents.device, latents.dtype) + bn_mean)
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]
