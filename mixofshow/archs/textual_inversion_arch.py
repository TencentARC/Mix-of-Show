import itertools
import os.path
import re
import torch
import torch.nn.functional as F

from mixofshow.archs.stable_diffusion_arch import Stable_Diffusion
from mixofshow.utils import get_root_logger
from mixofshow.utils.diffusers_sample_util import encode_text_feature
from mixofshow.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class Textual_Inversion(Stable_Diffusion):

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
        self.new_concept_token, self.new_concept_token_id = self.init_new_concept(new_concept_token, initializer_token)
        # 2. set freeze and optimize params
        if finetune_cfg:
            self.set_finetune_cfg(finetune_cfg)

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

        if 'new_concept_token' in delta_state_dict and len(delta_state_dict['new_concept_token']) != 0:
            new_concept_token = list(delta_state_dict['new_concept_token'].keys())

            # check whether new concept is initialized,
            if len(new_concept_token) == len(self.new_concept_token) and set(new_concept_token) == set(
                    self.new_concept_token):
                token_embeds = self.text_encoder.get_input_embeddings().weight.data
                for i, id_ in enumerate(self.new_concept_token_id):
                    logger.info(f'load: token_{id_} from {self.new_concept_token[i]}')
                    token_embeds[id_] = delta_state_dict['new_concept_token'][self.new_concept_token[i]]
            else:
                raise Exception('Your checkpoint have different concept with your model, please check')

    def delta_state_dict(self):
        delta_dict = {'new_concept_token': {}, 'text_encoder': {}, 'unet': {}}

        # save_embedding
        for i in range(len(self.new_concept_token_id)):
            learned_embeds = self.text_encoder.get_input_embeddings().weight[self.new_concept_token_id[i]]
            delta_dict['new_concept_token'][self.new_concept_token[i]] = learned_embeds.detach().cpu()

        return delta_dict
