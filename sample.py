import json
import os

import torch
from diffusers import DPMSolverMultistepScheduler

from mixofshow.pipelines.pipeline_edlora import EDLoRAPipeline

pretrained_model_path = 'experiments/composed_edlora/chilloutmix/potter+hermione+thanos_chilloutmix/combined_model_base'
enable_edlora = True  # True for edlora, False for lora

pipe = EDLoRAPipeline.from_pretrained(pretrained_model_path, scheduler=DPMSolverMultistepScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler'), torch_dtype=torch.float16).to('cuda')
with open(f'{pretrained_model_path}/new_concept_cfg.json', 'r') as fr:
    new_concept_cfg = json.load(fr)
pipe.set_new_concept_cfg(new_concept_cfg)

TOK = '<thanos1> <thanos2>, full body'  # the TOK is the concept name when training lora/edlora
prompt = f'a {TOK} in front of mount fuji'
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

image = pipe(prompt, negative_prompt=negative_prompt, height=1024, width=512, num_inference_steps=50, generator=torch.Generator('cuda').manual_seed(1), guidance_scale=7.5).images[0]

save_dir = 'debug_combined/combine_thanos/'
os.makedirs(save_dir, exist_ok=True)
image.save(f'{save_dir}/res.jpg')
