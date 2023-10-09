import itertools
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer

from mixofshow.models.edlora import (LoRALinearLayer, revise_edlora_unet_attention_controller_forward,
                                     revise_edlora_unet_attention_forward)
from mixofshow.pipelines.pipeline_edlora import bind_concept_prompt
from mixofshow.utils.ptp_util import AttentionStore


class EDLoRATrainer(nn.Module):
    def __init__(
        self,
        pretrained_path,
        new_concept_token,
        initializer_token,
        enable_edlora,  # true for ED-LoRA, false for LoRA
        finetune_cfg=None,
        noise_offset=None,
        attn_reg_weight=None,
        reg_full_identity=True,  # True for thanos, False for real person (don't need to encode clothes)
        use_mask_loss=True,
        enable_xformers=False,
        gradient_checkpoint=False
    ):
        super().__init__()

        # 1. Load the model.
        self.vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder='vae')
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_path, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder='text_encoder')
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder='unet')

        if gradient_checkpoint:
            self.unet.enable_gradient_checkpointing()

        if enable_xformers:
            assert is_xformers_available(), 'need to install xformer first'

        # 2. Define train scheduler
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_path, subfolder='scheduler')

        # 3. define training cfg
        self.enable_edlora = enable_edlora
        self.new_concept_cfg = self.init_new_concept(new_concept_token, initializer_token, enable_edlora=enable_edlora)

        self.attn_reg_weight = attn_reg_weight
        self.reg_full_identity = reg_full_identity
        if self.attn_reg_weight is not None:
            self.controller = AttentionStore(training=True)
            revise_edlora_unet_attention_controller_forward(self.unet, self.controller)  # support both lora and edlora forward
        else:
            revise_edlora_unet_attention_forward(self.unet)  # support both lora and edlora forward

        if finetune_cfg:
            self.set_finetune_cfg(finetune_cfg)

        self.noise_offset = noise_offset
        self.use_mask_loss = use_mask_loss

    def set_finetune_cfg(self, finetune_cfg):
        logger = get_logger('mixofshow', log_level='INFO')
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

        # 2. text encoder
        if finetune_cfg['text_encoder']['enable_tuning'] and finetune_cfg['text_encoder'].get('lora_cfg'):
            text_encoder_cfg = finetune_cfg['text_encoder']

            where = text_encoder_cfg['lora_cfg'].pop('where')
            assert where in ['CLIPEncoderLayer', 'CLIPAttention']

            self.text_encoder_lora = nn.ModuleList()
            params_list = []

            for name, module in self.text_encoder.named_modules():
                if module.__class__.__name__ == where:
                    for child_name, child_module in module.named_modules():
                        if child_module.__class__.__name__ == 'Linear':
                            lora_module = LoRALinearLayer(name + '.' + child_name, child_module, **text_encoder_cfg['lora_cfg'])
                            self.text_encoder_lora.append(lora_module)
                            params_list.extend(list(lora_module.parameters()))

            params_group_list.append({'params': params_list, 'lr': text_encoder_cfg['lr']})
            logger.info(f"optimizing text_encoder ({len(self.text_encoder_lora)} LoRAs), using lr: {text_encoder_cfg['lr']}")

        # 3. unet
        if finetune_cfg['unet']['enable_tuning'] and finetune_cfg['unet'].get('lora_cfg'):
            unet_cfg = finetune_cfg['unet']

            where = unet_cfg['lora_cfg'].pop('where')
            assert where in ['Transformer2DModel', 'Attention']

            self.unet_lora = nn.ModuleList()
            params_list = []

            for name, module in self.unet.named_modules():
                if module.__class__.__name__ == where:
                    for child_name, child_module in module.named_modules():
                        if child_module.__class__.__name__ == 'Linear' or (child_module.__class__.__name__ == 'Conv2d' and child_module.kernel_size == (1, 1)):
                            lora_module = LoRALinearLayer(name + '.' + child_name, child_module, **unet_cfg['lora_cfg'])
                            self.unet_lora.append(lora_module)
                            params_list.extend(list(lora_module.parameters()))

            params_group_list.append({'params': params_list, 'lr': unet_cfg['lr']})
            logger.info(f"optimizing unet ({len(self.unet_lora)} LoRAs), using lr: {unet_cfg['lr']}")

        # 4. optimize params
        self.params_to_optimize_iterator = params_group_list

    def get_params_to_optimize(self):
        return self.params_to_optimize_iterator

    def init_new_concept(self, new_concept_tokens, initializer_tokens, enable_edlora=True):
        logger = get_logger('mixofshow', log_level='INFO')
        new_concept_cfg = {}
        new_concept_tokens = new_concept_tokens.split('+')

        if initializer_tokens is None:
            initializer_tokens = ['<rand-0.017>'] * len(new_concept_tokens)
        else:
            initializer_tokens = initializer_tokens.split('+')
        assert len(new_concept_tokens) == len(initializer_tokens), 'concept token should match init token.'

        for idx, (concept_name, init_token) in enumerate(zip(new_concept_tokens, initializer_tokens)):
            if enable_edlora:
                num_new_embedding = 16
            else:
                num_new_embedding = 1
            new_token_names = [f'<new{idx * num_new_embedding + layer_id}>' for layer_id in range(num_new_embedding)]

            num_added_tokens = self.tokenizer.add_tokens(new_token_names)
            assert num_added_tokens == len(new_token_names), 'some token is already in tokenizer'
            new_token_ids = [self.tokenizer.convert_tokens_to_ids(token_name) for token_name in new_token_names]

            # init embedding
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            token_embeds = self.text_encoder.get_input_embeddings().weight.data

            if init_token.startswith('<rand'):
                sigma_val = float(re.findall(r'<rand-(.*)>', init_token)[0])
                init_feature = torch.randn_like(token_embeds[0]) * sigma_val
                logger.info(f'{concept_name} ({min(new_token_ids)}-{max(new_token_ids)}) is random initialized by: {init_token}')
            else:
                # Convert the initializer_token, placeholder_token to ids
                init_token_ids = self.tokenizer.encode(init_token, add_special_tokens=False)
                # print(token_ids)
                # Check if initializer_token is a single token or a sequence of tokens
                if len(init_token_ids) > 1 or init_token_ids[0] == 40497:
                    raise ValueError('The initializer token must be a single existing token.')
                init_feature = token_embeds[init_token_ids]
                logger.info(f'{concept_name} ({min(new_token_ids)}-{max(new_token_ids)}) is random initialized by existing token ({init_token}): {init_token_ids[0]}')

            for token_id in new_token_ids:
                token_embeds[token_id] = init_feature.clone()

            new_concept_cfg.update({
                concept_name: {
                    'concept_token_ids': new_token_ids,
                    'concept_token_names': new_token_names
                }
            })

        return new_concept_cfg

    def get_all_concept_token_ids(self):
        new_concept_token_ids = []
        for _, new_token_cfg in self.new_concept_cfg.items():
            new_concept_token_ids.extend(new_token_cfg['concept_token_ids'])
        return new_concept_token_ids

    def forward(self, images, prompts, masks, img_masks):
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

        if self.enable_edlora:
            prompts = bind_concept_prompt(prompts, new_concept_cfg=self.new_concept_cfg)  # edlora

        # get text ids
        text_input_ids = self.tokenizer(
            prompts,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt').input_ids.to(latents.device)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(text_input_ids)[0]
        if self.enable_edlora:
            encoder_hidden_states = rearrange(encoder_hidden_states, '(b n) m c -> b n m c', b=latents.shape[0])  # edlora

        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.scheduler.config.prediction_type == 'v_prediction':
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f'Unknown prediction type {self.scheduler.config.prediction_type}')

        if self.use_mask_loss:
            loss_mask = masks
        else:
            loss_mask = img_masks
        loss = F.mse_loss(model_pred.float(), target.float(), reduction='none')
        loss = ((loss * loss_mask).sum([1, 2, 3]) / loss_mask.sum([1, 2, 3])).mean()

        if self.attn_reg_weight is not None:
            attention_maps = self.controller.get_average_attention()
            attention_loss = self.cal_attn_reg(attention_maps, masks, text_input_ids)
            if not torch.isnan(attention_loss):  # full mask
                loss = loss + attention_loss
            self.controller.reset()

        return loss

    def cal_attn_reg(self, attention_maps, masks, text_input_ids):
        '''
        attention_maps: {down_cross:[], mid_cross:[], up_cross:[]}
        masks: torch.Size([1, 1, 64, 64])
        text_input_ids: torch.Size([16, 77])
        '''
        # step 1: find token position
        batch_size = masks.shape[0]
        text_input_ids = rearrange(text_input_ids, '(b l) n -> b l n', b=batch_size)
        # print(masks.shape) # torch.Size([2, 1, 64, 64])
        # print(text_input_ids.shape) # torch.Size([2, 16, 77])

        new_token_pos = []
        all_concept_token_ids = self.get_all_concept_token_ids()
        for text in text_input_ids:
            text = text[0]  # even multi-layer embedding, we extract the first one
            new_token_pos.append([idx for idx in range(len(text)) if text[idx] in all_concept_token_ids])

        # step2: aggregate attention maps with resolution and concat heads
        attention_groups = {'64': [], '32': [], '16': [], '8': []}
        for _, attention_list in attention_maps.items():
            for attn in attention_list:
                res = int(math.sqrt(attn.shape[1]))
                cross_map = attn.reshape(batch_size, -1, res, res, attn.shape[-1])
                attention_groups[str(res)].append(cross_map)

        for k, cross_map in attention_groups.items():
            cross_map = torch.cat(cross_map, dim=-4)  # concat heads
            cross_map = cross_map.sum(-4) / cross_map.shape[-4]  # e.g., 64 torch.Size([2, 64, 64, 77])
            cross_map = torch.stack([batch_map[..., batch_pos] for batch_pos, batch_map in zip(new_token_pos, cross_map)])  # torch.Size([2, 64, 64, 2])
            attention_groups[k] = cross_map

        attn_reg_total = 0
        # step3: calculate loss for each resolution: <new1> <new2> -> <new1> is to penalize outside mask, <new2> to align with mask
        for k, cross_map in attention_groups.items():
            map_adjective, map_subject = cross_map[..., 0], cross_map[..., 1]

            map_subject = map_subject / map_subject.max()
            map_adjective = map_adjective / map_adjective.max()

            gt_mask = F.interpolate(masks, size=map_subject.shape[1:], mode='nearest').squeeze(1)

            if self.reg_full_identity:
                loss_subject = F.mse_loss(map_subject.float(), gt_mask.float(), reduction='mean')
            else:
                loss_subject = map_subject[gt_mask == 0].mean()

            loss_adjective = map_adjective[gt_mask == 0].mean()

            attn_reg_total += self.attn_reg_weight * (loss_subject + loss_adjective)
        return attn_reg_total

    def load_delta_state_dict(self, delta_state_dict):
        # load embedding
        logger = get_logger('mixofshow', log_level='INFO')

        if 'new_concept_embedding' in delta_state_dict and len(delta_state_dict['new_concept_embedding']) != 0:
            new_concept_tokens = list(delta_state_dict['new_concept_embedding'].keys())

            # check whether new concept is initialized
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            if set(new_concept_tokens) != set(self.new_concept_cfg.keys()):
                logger.warning('Your checkpoint have different concept with your model, loading existing concepts')

            for concept_name, concept_cfg in self.new_concept_cfg.items():
                logger.info(f'load: concept_{concept_name}')
                token_embeds[concept_cfg['concept_token_ids']] = token_embeds[
                    concept_cfg['concept_token_ids']].copy_(delta_state_dict['new_concept_embedding'][concept_name])

        # load text_encoder
        if 'text_encoder' in delta_state_dict and len(delta_state_dict['text_encoder']) != 0:
            load_keys = delta_state_dict['text_encoder'].keys()
            if hasattr(self, 'text_encoder_lora') and len(load_keys) == 2 * len(self.text_encoder_lora):
                logger.info('loading LoRA for text encoder:')
                for lora_module in self.text_encoder_lora:
                    for name, param, in lora_module.named_parameters():
                        logger.info(f'load: {lora_module.name}.{name}')
                        param.data.copy_(delta_state_dict['text_encoder'][f'{lora_module.name}.{name}'])
            else:
                for name, param, in self.text_encoder.named_parameters():
                    if name in load_keys and 'token_embedding' not in name:
                        logger.info(f'load: {name}')
                        param.data.copy_(delta_state_dict['text_encoder'][f'{name}'])

        # load unet
        if 'unet' in delta_state_dict and len(delta_state_dict['unet']) != 0:
            load_keys = delta_state_dict['unet'].keys()
            if hasattr(self, 'unet_lora') and len(load_keys) == 2 * len(self.unet_lora):
                logger.info('loading LoRA for unet:')
                for lora_module in self.unet_lora:
                    for name, param, in lora_module.named_parameters():
                        logger.info(f'load: {lora_module.name}.{name}')
                        param.data.copy_(delta_state_dict['unet'][f'{lora_module.name}.{name}'])
            else:
                for name, param, in self.unet.named_parameters():
                    if name in load_keys:
                        logger.info(f'load: {name}')
                        param.data.copy_(delta_state_dict['unet'][f'{name}'])

    def delta_state_dict(self):
        delta_dict = {'new_concept_embedding': {}, 'text_encoder': {}, 'unet': {}}

        # save_embedding
        for concept_name, concept_cfg in self.new_concept_cfg.items():
            learned_embeds = self.text_encoder.get_input_embeddings().weight[concept_cfg['concept_token_ids']]
            delta_dict['new_concept_embedding'][concept_name] = learned_embeds.detach().cpu()

        # save text model
        for lora_module in self.text_encoder_lora:
            for name, param, in lora_module.named_parameters():
                delta_dict['text_encoder'][f'{lora_module.name}.{name}'] = param.cpu().clone()

        # save unet model
        for lora_module in self.unet_lora:
            for name, param, in lora_module.named_parameters():
                delta_dict['unet'][f'{lora_module.name}.{name}'] = param.cpu().clone()

        return delta_dict