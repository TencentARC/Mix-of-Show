import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

from mixofshow.utils.diffusers_sample_util import NEGATIVE_PROMPT, StableDiffusion_Sample
from mixofshow.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class Stable_Diffusion(nn.Module):

    def __init__(self, pretrained_path, clip_skip=None, sd_version='v1', test_sampler_type='ddim'):
        super().__init__()
        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder='vae')

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_path, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder='text_encoder')

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder='unet')

        # 4. Define scheduler
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_path, subfolder='scheduler')

        # 5. Define test scheduler
        if test_sampler_type == 'ddim':
            self.test_scheduler = DDIMScheduler.from_pretrained(pretrained_path, subfolder='scheduler')
        else:
            raise ValueError('Scheduler not supported')

        assert sd_version in ['v1', 'v2'], 'only support stable diffusion v1/v2'
        self.sd_version = sd_version

        self.validation_pipeline = StableDiffusionPipeline(
            tokenizer=self.tokenizer,
            vae=self.vae,
            text_encoder=self.text_encoder,
            unet=self.unet,
            scheduler=self.test_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False)
        self.clip_skip = clip_skip

    @torch.no_grad()
    def sample(self, prompt, latents=None, use_negative_prompt=False, num_inference_steps=50, guidance_scale=7.5):
        if self.sd_version == 'v1':
            height = 512
            width = 512
        elif self.sd_version == 'v2':
            height = 768
            width = 768
        else:
            raise NotImplementedError

        if use_negative_prompt:
            negative_prompt = [NEGATIVE_PROMPT] * len(prompt)
        else:
            negative_prompt = None

        images = StableDiffusion_Sample(
            self.validation_pipeline,
            prompt=prompt,
            clip_skip=self.clip_skip,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            latents=latents,
        ).images
        return images
