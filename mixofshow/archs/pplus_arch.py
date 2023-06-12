import itertools
import re
import torch
import torch.nn.functional as F
from einops import rearrange

from mixofshow.archs.edlora_override import revise_unet_attention_forward
from mixofshow.archs.stable_diffusion_arch import Stable_Diffusion
from mixofshow.utils import get_root_logger
from mixofshow.utils.diffusers_sample_util import NEGATIVE_PROMPT, StableDiffusion_PPlus_Sample, encode_text_feature
from mixofshow.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class PPlus(Stable_Diffusion):

    def __init__(self,
                 pretrained_path,
                 new_concept_token,
                 initializer_token,
                 finetune_cfg,
                 noise_offset=None,
                 clip_skip=None,
                 sd_version='v1',
                 test_sampler_type='ddim'):
        super().__init__(
            pretrained_path=pretrained_path,
            clip_skip=clip_skip,
            sd_version=sd_version,
            test_sampler_type=test_sampler_type)

        # 1. set placeholder token
        self.new_concept_cfg = self.init_new_concept(new_concept_token, initializer_token)

        # 2. set freeze and optimize params
        if finetune_cfg:
            self.set_finetune_cfg(finetune_cfg)
        else:
            revise_unet_attention_forward(self.unet)

        self.noise_offset = noise_offset

    def set_finetune_cfg(self, finetune_cfg):
        logger = get_root_logger()
        params_to_freeze = [self.vae.parameters(), self.text_encoder.parameters(), self.unet.parameters()]

        # step 1: close all parameters, required_grad to False
        for params in itertools.chain(*params_to_freeze):
            params.requires_grad = False

        # step 2: begin to add trainable paramters
        params_group_list = []

        # 1. text embedding
        if finetune_cfg['text_embedding']['enable_tuning']:
            text_embedding_cfg = finetune_cfg['text_embedding']

            params_list = []
            for params in self.text_encoder.get_input_embeddings().parameters():
                params.requires_grad = True
                params_list.append(params)

            params_group = {'params': params_list, 'lr': text_embedding_cfg['lr']}
            if 'weight_decay' in text_embedding_cfg:
                params_group.update({'weight_decay': text_embedding_cfg['weight_decay']})
            params_group_list.append(params_group)
            logger.info(f"optimizing embedding using lr: {text_embedding_cfg['lr']}")

        revise_unet_attention_forward(self.unet)
        # 4. optimize params
        self.params_to_optimize_iterator = params_group_list

    def get_params_to_optimize(self):
        return self.params_to_optimize_iterator

    def init_new_concept(self, new_concept_token, initializer_token):
        NUM_CROSS_ATTENTION_LAYERS = 16

        new_concept_cfg = {}
        new_concept_token = new_concept_token.split('+')

        if initializer_token is None:
            initializer_token = ['<rand-0.017>'] * len(new_concept_token)
        else:
            initializer_token = initializer_token.split('+')
        init_tokens = [(layer_id, initializer_token) for layer_id in range(NUM_CROSS_ATTENTION_LAYERS)]
        init_tokens = sorted(init_tokens, key=lambda k: k[0])

        logger = get_root_logger()
        for idx, concept_name in enumerate(new_concept_token):

            new_token_names = [
                f'<new{idx * NUM_CROSS_ATTENTION_LAYERS + layer_id}>' for layer_id in range(NUM_CROSS_ATTENTION_LAYERS)
            ]
            num_added_tokens = self.tokenizer.add_tokens(new_token_names)
            assert num_added_tokens == NUM_CROSS_ATTENTION_LAYERS
            new_token_ids = [self.tokenizer.convert_tokens_to_ids(token_name) for token_name in new_token_names]

            # init embedding
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            token_embeds = self.text_encoder.get_input_embeddings().weight.data

            # print(concept_name) # <tezuka1>
            for layer_id, token_id in enumerate(new_token_ids):
                _, init_token_layer = init_tokens[layer_id]
                init_token_layer = init_token_layer[idx]
                if init_token_layer.startswith('<rand'):
                    sigma_val = float(re.findall(r'<rand-(.*)>', init_token_layer)[0])
                    token_embeds[token_id] = torch.randn_like(token_embeds[0]) * sigma_val
                    logger.info(f'{token_id} is random initialized by: {init_token_layer}')
                else:
                    # Convert the initializer_token, placeholder_token to ids
                    init_token_ids = self.tokenizer.encode(init_token_layer, add_special_tokens=False)

                    # print(token_ids)
                    # Check if initializer_token is a single token or a sequence of tokens
                    if len(init_token_ids) > 1 or init_token_ids[0] == 40497:
                        raise ValueError('The initializer token must be a single existing token.')
                    token_embeds[token_id] = token_embeds[init_token_ids[0]].clone()
                    logger.info(f'{token_id} is random initialized by: {init_token_layer}, {init_token_ids[0]}')

            new_concept_cfg.update(
                {concept_name: {
                    'concept_token_ids': new_token_ids,
                    'concept_token_names': new_token_names
                }})

        return new_concept_cfg

    def get_all_concept_token_ids(self):
        new_concept_token_ids = []
        for concept_name, new_token_cfg in self.new_concept_cfg.items():
            new_concept_token_ids.extend(new_token_cfg['concept_token_ids'])
        return new_concept_token_ids

    def bind_concept_prompt(self, prompts):
        new_prompts = []
        for prompt in prompts:
            prompt = [prompt] * 16
            for concept_name, new_token_cfg in self.new_concept_cfg.items():
                prompt = [
                    p.replace(concept_name, new_name)
                    for p, new_name in zip(prompt, new_token_cfg['concept_token_names'])
                ]
            new_prompts.extend(prompt)
        return new_prompts

    def forward(self, images, prompts, masks):
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)

        if self.noise_offset is not None:
            noise += self.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)

        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz, ), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        prompts = self.bind_concept_prompt(prompts)

        # get text ids
        text_input_ids = self.tokenizer(
            prompts,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt').input_ids.to(latents.device)

        # Get the text embedding for conditioning
        encoder_hidden_states = encode_text_feature(text_input_ids, self.text_encoder, clip_skip=self.clip_skip)
        encoder_hidden_states = rearrange(encoder_hidden_states, '(b n) m c -> b n m c', b=latents.shape[0])
        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.scheduler.config.prediction_type == 'v_prediction':
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f'Unknown prediction type {self.scheduler.config.prediction_type}')

        loss = F.mse_loss(model_pred.float(), target.float(), reduction='none')
        loss = ((loss * masks).sum([1, 2, 3]) / masks.sum([1, 2, 3])).mean()
        return loss

    def load_delta_state_dict(self, delta_state_dict):
        # load embedding
        logger = get_root_logger()

        if 'new_concept_embedding' in delta_state_dict and len(delta_state_dict['new_concept_embedding']) != 0:
            new_concept_tokens = list(delta_state_dict['new_concept_embedding'].keys())

            # check whether new concept is initialized
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            if set(new_concept_tokens) == set(self.new_concept_cfg.keys()):
                for concept_name, concept_cfg in self.new_concept_cfg.items():
                    logger.info(f'load: concept_{concept_name}')
                    token_embeds[concept_cfg['concept_token_ids']] = token_embeds[
                        concept_cfg['concept_token_ids']].copy_(delta_state_dict['new_concept_embedding'][concept_name])
            else:
                raise Exception('Your checkpoint have different concept with your model, please check')

    def delta_state_dict(self):
        delta_dict = {'new_concept_embedding': {}, 'text_encoder': {}, 'unet': {}}

        # save_embedding
        for concept_name, concept_cfg in self.new_concept_cfg.items():
            learned_embeds = self.text_encoder.get_input_embeddings().weight[concept_cfg['concept_token_ids']]
            delta_dict['new_concept_embedding'][concept_name] = learned_embeds.detach().cpu()

        return delta_dict

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

        images = StableDiffusion_PPlus_Sample(
            self.validation_pipeline,
            prompt=prompt,
            height=height,
            width=width,
            clip_skip=self.clip_skip,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            latents=latents,
            new_concept_cfg=self.new_concept_cfg).images
        return images
