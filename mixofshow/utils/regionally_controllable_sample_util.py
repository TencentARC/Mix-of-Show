import math
import numpy as np
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from einops import rearrange
from PIL import Image
from PIL import Image as PIL_Image
from torch import einsum
from typing import Any, Callable, Dict, List, Optional, Union

from mixofshow.utils.diffusers_sample_util import bind_concept_prompt


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL_Image.Image):
        image = [image]

    if isinstance(image[0], PIL_Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=Image.LANCZOS))[None, :] for i in image]
        image = np.concatenate(image, axis=0)

        if len(image.shape) == 3:
            image = image[..., None]

        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        # image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


def register_region_aware_attention(model, layerwise_embedding=False):

    def get_new_forward(cross_attention_idx):

        def region_rewrite(self, hidden_states, query, region_list, height, width):

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

                if self.upcast_attention:
                    query = query.float()
                    region_key = region_key.float()

                start_h, start_w, end_h, end_w = region_box
                start_h, start_w, end_h, end_w = math.ceil(start_h * feat_height), math.ceil(
                    start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)

                attention_region = einsum('b h w c, b n c -> b h w n', query[:, start_h:end_h, start_w:end_w, :],
                                          region_key) * self.scale
                if self.upcast_softmax:
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

        def new_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                is_cross = False
                encoder_hidden_states = hidden_states
            else:
                is_cross = True
                if layerwise_embedding:
                    encoder_hidden_states = encoder_hidden_states[:, cross_attention_idx, ...]
                else:
                    encoder_hidden_states = encoder_hidden_states

            if self.cross_attention_norm:
                encoder_hidden_states = self.norm_cross(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)

            hidden_states = torch.bmm(attention_probs, value)

            if is_cross:
                region_list = []
                for region in cross_attention_kwargs['region_list']:
                    if layerwise_embedding:
                        region_key = self.to_k(region[0][:, cross_attention_idx, ...])
                        region_value = self.to_v(region[0][:, cross_attention_idx, ...])
                    else:
                        region_key = self.to_k(region[0])
                        region_value = self.to_v(region[0])
                    region_key = self.head_to_batch_dim(region_key)
                    region_value = self.head_to_batch_dim(region_value)
                    region_list.append((region_key, region_value, region[1]))

                hidden_states = region_rewrite(
                    self,
                    hidden_states=hidden_states,
                    query=query,
                    region_list=region_list,
                    height=cross_attention_kwargs['height'],
                    width=cross_attention_kwargs['width'])

            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states

        return new_forward

    def change_forward(unet, cross_attention_idx):  # omit proceesor in new diffusers
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'CrossAttention':
                bound_method = get_new_forward(cross_attention_idx).__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
                if name == 'attn2':
                    cross_attention_idx += 1
            else:
                cross_attention_idx = change_forward(layer, cross_attention_idx)
        return cross_attention_idx

    # use this to ensure the order
    cross_attention_idx = change_forward(model.unet.down_blocks, 0)
    cross_attention_idx = change_forward(model.unet.mid_block, cross_attention_idx)
    _ = change_forward(model.unet.up_blocks, cross_attention_idx)


def encode_region_prompt_pplus(self,
                               prompt,
                               new_concept_cfg,
                               device,
                               num_images_per_prompt,
                               do_classifier_free_guidance,
                               negative_prompt=None,
                               prompt_embeds: Optional[torch.FloatTensor] = None,
                               negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                               height=512,
                               width=512):
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
def Regionally_T2IAdaptor_Sample(self,
                                 prompt: Union[str, List[str]] = None,
                                 new_concept_cfg=None,
                                 keypose_adapter_input: Union[torch.Tensor, PIL_Image.Image,
                                                              List[PIL_Image.Image]] = None,
                                 keypose_adaptor_weight=1.0,
                                 region_keypose_adaptor_weight='',
                                 sketch_adapter_input: Union[torch.Tensor, PIL_Image.Image,
                                                             List[PIL_Image.Image]] = None,
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
                                 cross_attention_kwargs: Optional[Dict[str, Any]] = None):

    if new_concept_cfg is None:
        # register region aware attention for sd embedding
        register_region_aware_attention(self, layerwise_embedding=False)
    else:
        # register region aware attention for pplus embedding
        register_region_aware_attention(self, layerwise_embedding=True)

    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

    if keypose_adapter_input is not None:
        keypose_input = preprocess(keypose_adapter_input).to(self.device)
        keypose_input = keypose_input.to(self.keypose_adapter.dtype)
    else:
        keypose_input = None

    if sketch_adapter_input is not None:
        sketch_input = preprocess(sketch_adapter_input).to(self.device)
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

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds, region_list = encode_region_prompt_pplus(
        self,
        prompt,
        new_concept_cfg,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        height=height,
        width=width)

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
    if keypose_input is not None:
        keypose_adapter_state = self.keypose_adapter(keypose_input)
        keys = keypose_adapter_state.keys()
    else:
        keypose_adapter_state = None

    if sketch_input is not None:
        sketch_adapter_state = self.sketch_adapter(sketch_input)
        keys = sketch_adapter_state.keys()
    else:
        sketch_adapter_state = None

    adapter_state = keypose_adapter_state if keypose_adapter_state is not None else sketch_adapter_state

    if do_classifier_free_guidance:
        for k in keys:
            if keypose_adapter_state is not None:
                feat_keypose = keypose_adapter_state[k]
                # print(feat_keypose.shape) # torch.Size([1, 320, 64, 128])
                spatial_adaptor_weight = keypose_adaptor_weight * torch.ones(*feat_keypose.shape[2:]).to(
                    feat_keypose.dtype).to(feat_keypose.device)

                if region_keypose_adaptor_weight != '':
                    keypose_region_list = region_keypose_adaptor_weight.split('|')

                    for region_weight in keypose_region_list:
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
                feat_sketch = sketch_adapter_state[k]
                # print(feat_keypose.shape) # torch.Size([1, 320, 64, 128])
                spatial_adaptor_weight = sketch_adaptor_weight * torch.ones(*feat_sketch.shape[2:]).to(
                    feat_sketch.dtype).to(feat_sketch.device)

                if region_sketch_adaptor_weight != '':
                    sketch_region_list = region_sketch_adaptor_weight.split('|')

                    for region_weight in sketch_region_list:
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

            adapter_state[k] = torch.cat([feat_keypose + feat_sketch] * 2, dim=0)

    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            self.unet.sideload_processor.update_sideload(adapter_state)
            # predict the noise residual

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs={
                    'region_list': region_list,
                    'height': height,
                    'width': width,
                }
                # downsample_adapter_states=adapter_dw_state,
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

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
