from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers import StableDiffusionPipeline
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate
from einops import rearrange
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from mixofshow.models.edlora import (revise_edlora_unet_attention_controller_forward,
                                     revise_edlora_unet_attention_forward)


def bind_concept_prompt(prompts, new_concept_cfg):
    if isinstance(prompts, str):
        prompts = [prompts]
    new_prompts = []
    for prompt in prompts:
        prompt = [prompt] * 16
        for concept_name, new_token_cfg in new_concept_cfg.items():
            prompt = [
                p.replace(concept_name, new_name) for p, new_name in zip(prompt, new_token_cfg['concept_token_names'])
            ]
        new_prompts.extend(prompt)
    return new_prompts


class EDLoRAPipeline(StableDiffusionPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker: bool = False,
    ):
        if hasattr(scheduler.config, 'steps_offset') and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f'The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`'
                f' should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure '
                'to update the config accordingly as leaving `steps_offset` might led to incorrect results'
                ' in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,'
                ' it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`'
                ' file'
            )
            deprecate('steps_offset!=1', '1.0.0', deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config['steps_offset'] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, 'clip_sample') and scheduler.config.clip_sample is True:
            deprecation_message = (
                f'The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`.'
                ' `clip_sample` should be set to False in the configuration file. Please make sure to update the'
                ' config accordingly as not setting `clip_sample` in the config might lead to incorrect results in'
                ' future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very'
                ' nice if you could open a Pull request for the `scheduler/scheduler_config.json` file'
            )
            deprecate('clip_sample not set', '1.0.0', deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config['clip_sample'] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, '_diffusers_version') and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse('0.9.0.dev0')
        is_unet_sample_size_less_64 = hasattr(unet.config, 'sample_size') and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                'The configuration file of the unet has set the default `sample_size` to smaller than'
                ' 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the'
                ' following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-'
                ' CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5'
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                ' configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`'
                ' in the config might lead to incorrect results in future versions. If you have downloaded this'
                ' checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for'
                ' the `unet/config.json` file'
            )
            deprecate('sample_size<64', '1.0.0', deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config['sample_size'] = 64
            unet._internal_dict = FrozenDict(new_config)

        revise_edlora_unet_attention_forward(unet)
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.new_concept_cfg = None

    def set_new_concept_cfg(self, new_concept_cfg=None):
        self.new_concept_cfg = new_concept_cfg

    def set_controller(self, controller):
        self.controller = controller
        revise_edlora_unet_attention_controller_forward(self.unet, controller)

    def _encode_prompt(self,
        prompt,
        new_concept_cfg,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None
    ):

        assert num_images_per_prompt == 1, 'only support num_images_per_prompt=1 now'

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:

            prompt_extend = bind_concept_prompt(prompt, new_concept_cfg)

            text_inputs = self.tokenizer(
                prompt_extend,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt',
            )
            text_input_ids = text_inputs.input_ids

            prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
            prompt_embeds = rearrange(prompt_embeds, '(b n) m c -> b n m c', b=batch_size)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, layer_num, seq_len, _ = prompt_embeds.shape

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [''] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !='
                                f' {type(prompt)}.')
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:'
                    f' {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches'
                    ' the batch size of `prompt`.')
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding='max_length',
                max_length=seq_len,
                truncation=True,
                return_tensors='pt',
            )

            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(device))[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            negative_prompt_embeds = (negative_prompt_embeds).view(batch_size, 1, seq_len, -1).repeat(1, layer_num, 1, 1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt, this support pplus and edlora (layer-wise embedding)
        assert self.new_concept_cfg is not None
        prompt_embeds = self._encode_prompt(
            prompt,
            self.new_concept_cfg,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if hasattr(self, 'controller'):
                    dtype = latents.dtype
                    latents = self.controller.step_callback(latents)
                    latents = latents.to(dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == 'latent':
            image = latents
        elif output_type == 'pil':
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

        # Offload last model to CPU
        if hasattr(self, 'final_offload_hook') and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
