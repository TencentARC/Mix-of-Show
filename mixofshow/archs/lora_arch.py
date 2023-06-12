import itertools
import os.path
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from mixofshow.archs.lora_override import LoRALinearLayer, lora_moved
from mixofshow.archs.stable_diffusion_arch import Stable_Diffusion
from mixofshow.utils import get_root_logger
from mixofshow.utils.diffusers_sample_util import encode_text_feature
from mixofshow.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class LoRA(Stable_Diffusion):

    def __init__(self,
                 pretrained_path,
                 new_concept_token,
                 initializer_token,
                 finetune_cfg,
                 clip_skip=None,
                 sd_version='v1',
                 test_sampler_type='ddim'):
        super().__init__(
            pretrained_path=pretrained_path,
            sd_version=sd_version,
            clip_skip=clip_skip,
            test_sampler_type=test_sampler_type)

        # 1. set placeholder token, get the placeholder token list and index list and change the model accordingly
        if new_concept_token is not None:
            self.new_concept_token, self.new_concept_token_id = self.init_new_concept(
                new_concept_token, initializer_token)
        else:
            self.new_concept_token, self.new_concept_token_id = [], []

        # 2. set freeze and optimize params
        if finetune_cfg:
            self.init_colora(finetune_cfg)

    def init_colora(self, finetune_cfg):
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
                            lora_module = LoRALinearLayer(name + '.' + child_name + '_lora', child_module,
                                                          **text_encoder_cfg['lora_cfg'])
                            self.text_encoder_lora.append(lora_module)
                            params_list.extend(list(lora_module.parameters()))

            params_group_list.append({'params': params_list, 'lr': text_encoder_cfg['lr']})
            logger.info(
                f"optimizing text_encoder ({len(self.text_encoder_lora)} LoRAs), using lr: {text_encoder_cfg['lr']}")

        # 3. cross-attention
        if finetune_cfg['unet']['enable_tuning'] and finetune_cfg['unet'].get('lora_cfg'):
            unet_cfg = finetune_cfg['unet']

            where = unet_cfg['lora_cfg'].pop('where')
            assert where in ['Transformer2DModel', 'CrossAttention']

            self.unet_lora = nn.ModuleList()
            params_list = []

            for name, module in self.unet.named_modules():
                if module.__class__.__name__ == where:
                    for child_name, child_module in module.named_modules():
                        if child_module.__class__.__name__ == 'Linear' or (child_module.__class__.__name__ == 'Conv2d'
                                                                           and child_module.kernel_size == (1, 1)):
                            lora_module = LoRALinearLayer(name + '.' + child_name + '_lora', child_module,
                                                          **unet_cfg['lora_cfg'])
                            self.unet_lora.append(lora_module)
                            params_list.extend(list(lora_module.parameters()))

            params_group_list.append({'params': params_list, 'lr': unet_cfg['lr']})
            logger.info(f"optimizing unet ({len(self.unet_lora)} LoRAs), using lr: {unet_cfg['lr']}")

        # 4. optimize params
        self.params_to_optimize_iterator = params_group_list

    def get_params_to_optimize(self):
        return self.params_to_optimize_iterator

    def init_new_concept(self, new_concept_tokens, initializer_tokens):
        '''
        Args:
            new_concept_token: <new1>+<new2>+...
            initializer_token: token 1+ token 2+..., use those embedding to initialize new_concept tokens
        Returns:
            new_concept_token: [<new1>, <new2>, ...]
            new_concept_token_id: [id1, id2]
        '''

        logger = get_root_logger()

        new_concept_token_ids = []
        new_concept_tokens = new_concept_tokens.split('+')

        if initializer_tokens is None:
            initializer_token = ['<rand-0.017>'] * len(new_concept_tokens)
        else:
            initializer_token = initializer_tokens.split('+')
        assert len(new_concept_tokens) == len(initializer_token), 'concept token should match init token.'

        for new_token, init_token in zip(new_concept_tokens, initializer_token):
            # Add the placeholder token in tokenizer
            num_added_tokens = self.tokenizer.add_tokens(new_token)
            if num_added_tokens == 0:
                raise ValueError(f'The tokenizer already contains the token {new_token}. Please pass a different'
                                 ' `new_concept_token` that is not already in the tokenizer.')

            new_token_id = self.tokenizer.convert_tokens_to_ids(new_token)
            new_concept_token_ids.append(new_token_id)

            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            token_embeds = self.text_encoder.get_input_embeddings().weight.data

            # init_token may be filepath
            if init_token.startswith('<rand'):
                sigma_val = float(re.findall(r'<rand-(.*)>', init_token)[0])
                token_embeds[new_token_id] = torch.randn_like(token_embeds[0]) * sigma_val
                logger.info(f'{new_token} is random initialized by: {init_token}')
            elif os.path.exists(init_token):
                token_feature = torch.load(init_token)
                token_embeds[new_token_id] = token_feature
                logger.info(f'{new_token} is initialized by pretrained: {init_token}')
            else:
                # Convert the initializer_token, placeholder_token to ids
                init_token_ids = self.tokenizer.encode(init_token, add_special_tokens=False)
                # print(token_ids)
                # Check if initializer_token is a single token or a sequence of tokens
                if len(init_token_ids) > 1 or init_token_ids[0] == 40497:
                    raise ValueError('The initializer token must be a single existing token.')
                token_embeds[new_token_id] = token_embeds[init_token_ids[0]]
                logger.info(f'{new_token} is initialized by existing token ({init_token}): {init_token_ids[0]}')

        return new_concept_tokens, new_concept_token_ids

    def forward(self, images, prompts, masks):
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz, ), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # get text ids
        text_input_ids = self.tokenizer(
            prompts,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt').input_ids.to(latents.device)

        # Get the text embedding for conditioning
        encoder_hidden_states = encode_text_feature(text_input_ids, self.text_encoder, clip_skip=self.clip_skip)

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
            new_concept_token = list(delta_state_dict['new_concept_embedding'].keys())

            # check whether new concept is initialized,
            if len(new_concept_token) != len(self.new_concept_token):
                logger.warning('Your checkpoint have different concept with your model, loading existing concepts')

            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            for i, id_ in enumerate(self.new_concept_token_id):
                logger.info(f'load: token_{id_} from {self.new_concept_token[i]}')
                token_embeds[id_] = delta_state_dict['new_concept_embedding'][self.new_concept_token[i]]

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

        logger = get_root_logger()

        # save_embedding
        for i in range(len(self.new_concept_token_id)):
            learned_embeds = self.text_encoder.get_input_embeddings().weight[self.new_concept_token_id[i]]
            delta_dict['new_concept_embedding'][self.new_concept_token[i]] = learned_embeds.detach().cpu()

        # save text model
        for lora_module in self.text_encoder_lora:
            for name, param, in lora_module.named_parameters():
                delta_dict['text_encoder'][f'{lora_module.name}.{name}'] = param.cpu().clone()

        if len(delta_dict['text_encoder']) != 0:
            text_encoder_moved = lora_moved(delta_dict['text_encoder'])
            logger.info(f'text_encoder moved: {text_encoder_moved}')

        # save unet model
        for lora_module in self.unet_lora:
            for name, param, in lora_module.named_parameters():
                delta_dict['unet'][f'{lora_module.name}.{name}'] = param.cpu().clone()
        if len(delta_dict['unet']) != 0:
            unet_moved = lora_moved(delta_dict['unet'])
            logger.info(f'unet moved: {unet_moved}')

        return delta_dict
