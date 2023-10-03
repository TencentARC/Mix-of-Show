import math
from typing import Any, Callable, Dict, List, Optional, Union

import PIL
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.t2i_adapter.pipeline_stable_diffusion_adapter import (StableDiffusionAdapterPipeline,
                                                                               StableDiffusionAdapterPipelineOutput,
                                                                               _preprocess_adapter_image)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from torch import einsum
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

if is_xformers_available():
    import xformers

from mixofshow.pipelines.pipeline_edlora import bind_concept_prompt

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class RegionT2I_AttnProcessor:
    def __init__(self, cross_attention_idx, attention_op=None):
        self.attention_op = attention_op
        self.cross_attention_idx = cross_attention_idx

    def region_rewrite(self, attn, hidden_states, query, region_list, height, width):

        def get_region_mask(region_list, feat_height, feat_width):
            exclusive_mask = torch.zeros((feat_height, feat_width))
            for region in region_list:
                start_h, start_w, end_h, end_w = region[-1]
                start_h, start_w, end_h, end_w = math.ceil(start_h * feat_height), math.ceil(
                    start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)
                exclusive_mask[start_h:end_h, start_w:end_w] += 1
            return exclusive_mask

        dtype = query.dtype
        seq_lens = query.shape[1]
        downscale = math.sqrt(height * width / seq_lens)

        # 0: context >=1: may be overlap
        feat_height, feat_width = int(height // downscale), int(width // downscale)
        region_mask = get_region_mask(region_list, feat_height, feat_width)

        query = rearrange(query, 'b (h w) c -> b h w c', h=feat_height, w=feat_width)
        hidden_states = rearrange(hidden_states, 'b (h w) c -> b h w c', h=feat_height, w=feat_width)

        new_hidden_state = torch.zeros_like(hidden_states)
        new_hidden_state[:, region_mask == 0, :] = hidden_states[:, region_mask == 0, :]

        replace_ratio = 1.0
        new_hidden_state[:, region_mask != 0, :] = (1 - replace_ratio) * hidden_states[:, region_mask != 0, :]

        for region in region_list:
            region_key, region_value, region_box = region

            if attn.upcast_attention:
                query = query.float()
                region_key = region_key.float()

            start_h, start_w, end_h, end_w = region_box
            start_h, start_w, end_h, end_w = math.ceil(start_h * feat_height), math.ceil(
                start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)

            attention_region = einsum('b h w c, b n c -> b h w n', query[:, start_h:end_h, start_w:end_w, :], region_key) * attn.scale
            if attn.upcast_softmax:
                attention_region = attention_region.float()

            attention_region = attention_region.softmax(dim=-1)
            attention_region = attention_region.to(dtype)

            hidden_state_region = einsum('b h w n, b n c -> b h w c', attention_region, region_value)
            new_hidden_state[:, start_h:end_h, start_w:end_w, :] += \
                replace_ratio * (hidden_state_region / (
                    region_mask.reshape(
                        1, *region_mask.shape, 1)[:, start_h:end_h, start_w:end_w, :]
                ).to(query.device))

        new_hidden_state = rearrange(new_hidden_state, 'b h w c -> b (h w) c')
        return new_hidden_state

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **cross_attention_kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            is_cross = False
            encoder_hidden_states = hidden_states
        else:
            is_cross = True

            if len(encoder_hidden_states.shape) == 4:  # multi-layer embedding
                encoder_hidden_states = encoder_hidden_states[:, self.cross_attention_idx, ...]
            else:
                encoder_hidden_states = encoder_hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if is_xformers_available() and not is_cross:
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)

        if is_cross:
            region_list = []
            for region in cross_attention_kwargs['region_list']:
                if len(region[0].shape) == 4:
                    region_key = attn.to_k(region[0][:, self.cross_attention_idx, ...])
                    region_value = attn.to_v(region[0][:, self.cross_attention_idx, ...])
                else:
                    region_key = attn.to_k(region[0])
                    region_value = attn.to_v(region[0])
                region_key = attn.head_to_batch_dim(region_key)
                region_value = attn.head_to_batch_dim(region_value)
                region_list.append((region_key, region_value, region[1]))

            hidden_states = self.region_rewrite(
                attn=attn,
                hidden_states=hidden_states,
                query=query,
                region_list=region_list,
                height=cross_attention_kwargs['height'],
                width=cross_attention_kwargs['width'])

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def revise_regionally_t2iadapter_attention_forward(unet):
    def change_forward(unet, count):
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention':
                layer.set_processor(RegionT2I_AttnProcessor(count))
                if 'attn2' in name:
                    count += 1
            else:
                count = change_forward(layer, count)
        return count

    # use this to ensure the order
    cross_attention_idx = change_forward(unet.down_blocks, 0)
    cross_attention_idx = change_forward(unet.mid_block, cross_attention_idx)
    cross_attention_idx = change_forward(unet.up_blocks, cross_attention_idx)
    print(f'Number of attention layer registered {cross_attention_idx}')


class RegionallyT2IAdapterPipeline(StableDiffusionAdapterPipeline):
    _optional_components = ['safety_checker', 'feature_extractor']

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = False,
    ):

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f'You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure'
                ' that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered'
                ' results in services or applications open to the public. Both the diffusers team and Hugging Face'
                ' strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling'
                ' it only for use-cases that involve analyzing network behavior or auditing its results. For more'
                ' information, please have a look at https://github.com/huggingface/diffusers/pull/254 .'
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                'Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety'
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.new_concept_cfg = None
        revise_regionally_t2iadapter_attention_forward(self.unet)

    def set_new_concept_cfg(self, new_concept_cfg=None):
        self.new_concept_cfg = new_concept_cfg

    def _encode_region_prompt(self,
        prompt,
        new_concept_cfg,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        height=512,
        width=512
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        assert batch_size == 1, 'only sample one prompt once in this version'

        if prompt_embeds is None:
            context_prompt, region_list = prompt[0][0], prompt[0][1]
            context_prompt = bind_concept_prompt([context_prompt], new_concept_cfg)
            context_prompt_input_ids = self.tokenizer(
                context_prompt,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt',
            ).input_ids

            prompt_embeds = self.text_encoder(context_prompt_input_ids.to(device), attention_mask=None)[0]
            prompt_embeds = rearrange(prompt_embeds, '(b n) m c -> b n m c', b=batch_size)
            prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            bs_embed, layer_num, seq_len, _ = prompt_embeds.shape

            if negative_prompt is None:
                negative_prompt = [''] * batch_size

            negative_prompt_input_ids = self.tokenizer(
                negative_prompt,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt').input_ids

            negative_prompt_embeds = self.text_encoder(
                negative_prompt_input_ids.to(device),
                attention_mask=None,
            )[0]

            negative_prompt_embeds = (negative_prompt_embeds).view(batch_size, 1, seq_len, -1).repeat(1, layer_num, 1, 1)
            negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            for idx, region in enumerate(region_list):
                region_prompt, region_neg_prompt, pos = region
                region_prompt = bind_concept_prompt([region_prompt], new_concept_cfg)
                region_prompt_input_ids = self.tokenizer(
                    region_prompt,
                    padding='max_length',
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors='pt').input_ids
                region_embeds = self.text_encoder(region_prompt_input_ids.to(device), attention_mask=None)[0]
                region_embeds = rearrange(region_embeds, '(b n) m c -> b n m c', b=batch_size)
                region_embeds.to(dtype=self.text_encoder.dtype, device=device)
                bs_embed, layer_num, seq_len, _ = region_embeds.shape

                if region_neg_prompt is None:
                    region_neg_prompt = [''] * batch_size
                region_negprompt_input_ids = self.tokenizer(
                    region_neg_prompt,
                    padding='max_length',
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors='pt').input_ids
                region_neg_embeds = self.text_encoder(region_negprompt_input_ids.to(device), attention_mask=None)[0]
                region_neg_embeds = (region_neg_embeds).view(batch_size, 1, seq_len, -1).repeat(1, layer_num, 1, 1)
                region_neg_embeds.to(dtype=self.text_encoder.dtype, device=device)
                region_list[idx] = (torch.cat([region_neg_embeds, region_embeds]), pos)

        return prompt_embeds, region_list

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        keypose_adapter_input: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
        keypose_adaptor_weight=1.0,
        region_keypose_adaptor_weight='',
        sketch_adapter_input: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
        sketch_adaptor_weight=1.0,
        region_sketch_adaptor_weight='',
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
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `List[torch.FloatTensor]` or `List[PIL.Image.Image]` or `List[List[PIL.Image.Image]]`):
                The Adapter input condition. Adapter uses this input condition to generate guidance to Unet. If the
                type is specified as `Torch.FloatTensor`, it is passed to Adapter as is. PIL.Image.Image` can also be
                accepted as an image. The control image is automatically resized to fit the output image.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionAdapterPipelineOutput`] instead
                of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            adapter_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the adapter are multiplied by `adapter_conditioning_scale` before they are added to the
                residual in the original unet. If multiple adapters are specified in init, you can set the
                corresponding scale as a list.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionAdapterPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionAdapterPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple. When returning a tuple, the first element is a list with the generated images, and the second
            element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        device = self._execution_device

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        if keypose_adapter_input is not None:
            keypose_input = _preprocess_adapter_image(keypose_adapter_input, height, width).to(self.device)
            keypose_input = keypose_input.to(self.keypose_adapter.dtype)
        else:
            keypose_input = None

        if sketch_adapter_input is not None:
            sketch_input = _preprocess_adapter_image(sketch_adapter_input, height, width).to(self.device)
            sketch_input = sketch_input.to(self.sketch_adapter.dtype)
        else:
            sketch_input = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        assert self.new_concept_cfg is not None
        prompt_embeds, region_list = self._encode_region_prompt(
            prompt,
            self.new_concept_cfg,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=height,
            width=width
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
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
        if keypose_input is not None:
            keypose_adapter_state = self.keypose_adapter(keypose_input)
        else:
            keypose_adapter_state = None

        if sketch_input is not None:
            sketch_adapter_state = self.sketch_adapter(sketch_input)
        else:
            sketch_adapter_state = None

        num_states = len(keypose_adapter_state) if keypose_adapter_state is not None else len(sketch_adapter_state)

        adapter_state = []

        for idx in range(num_states):
            if keypose_adapter_state is not None:
                feat_keypose = keypose_adapter_state[idx]

                spatial_adaptor_weight = keypose_adaptor_weight * torch.ones(*feat_keypose.shape[2:]).to(
                    feat_keypose.dtype).to(feat_keypose.device)

                if region_keypose_adaptor_weight != '':
                    region_list = region_keypose_adaptor_weight.split('|')

                    for region_weight in region_list:
                        region, weight = region_weight.split('-')
                        region = eval(region)
                        weight = eval(weight)
                        feat_height, feat_width = feat_keypose.shape[2:]
                        start_h, start_w, end_h, end_w = region
                        start_h, end_h = start_h / height, end_h / height
                        start_w, end_w = start_w / width, end_w / width

                        start_h, start_w, end_h, end_w = math.ceil(start_h * feat_height), math.ceil(
                            start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)

                        spatial_adaptor_weight[start_h:end_h, start_w:end_w] = weight
                feat_keypose = spatial_adaptor_weight * feat_keypose

            else:
                feat_keypose = 0

            if sketch_adapter_state is not None:
                feat_sketch = sketch_adapter_state[idx]
                # print(feat_keypose.shape) # torch.Size([1, 320, 64, 128])
                spatial_adaptor_weight = sketch_adaptor_weight * torch.ones(*feat_sketch.shape[2:]).to(
                    feat_sketch.dtype).to(feat_sketch.device)

                if region_sketch_adaptor_weight != '':
                    region_list = region_sketch_adaptor_weight.split('|')

                    for region_weight in region_list:
                        region, weight = region_weight.split('-')
                        region = eval(region)
                        weight = eval(weight)
                        feat_height, feat_width = feat_sketch.shape[2:]
                        start_h, start_w, end_h, end_w = region
                        start_h, end_h = start_h / height, end_h / height
                        start_w, end_w = start_w / width, end_w / width

                        start_h, start_w, end_h, end_w = math.ceil(start_h * feat_height), math.ceil(
                            start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)

                        spatial_adaptor_weight[start_h:end_h, start_w:end_w] = weight
                feat_sketch = spatial_adaptor_weight * feat_sketch
            else:
                feat_sketch = 0

            adapter_state.append(feat_keypose + feat_sketch)

        if do_classifier_free_guidance:
            for k, v in enumerate(adapter_state):
                adapter_state[k] = torch.cat([v] * 2, dim=0)

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
                    cross_attention_kwargs={
                        'region_list': region_list,
                        'height': height,
                        'width': width,
                    },
                    down_block_additional_residuals=[state.clone() for state in adapter_state],
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == 'latent':
            image = latents
            has_nsfw_concept = None
        elif output_type == 'pil':
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, 'final_offload_hook') and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionAdapterPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
