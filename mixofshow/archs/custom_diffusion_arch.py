import itertools
import torch
import torch.nn.functional as F
from diffusers.models.attention import CrossAttention

from mixofshow.archs.stable_diffusion_arch import Stable_Diffusion
from mixofshow.utils import get_root_logger
from mixofshow.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class Custom_Diffusion(Stable_Diffusion):

    def __init__(self,
                 pretrained_path,
                 modifier_token,
                 initializer_token,
                 finetune_params='crossattn_kv',
                 sd_version='v1',
                 test_sampler_type='ddim',
                 prior_loss_weight=None):
        super().__init__(pretrained_path=pretrained_path, sd_version=sd_version, test_sampler_type=test_sampler_type)

        # 1. set which params to finetune and fix others
        self.finetune_params = finetune_params
        self.create_custom_diffusion()

        # 2. set modifier token, get the modifier token list and index list and change the model accordingly
        self.modifier_token, self.modifier_token_id = self.init_new_concept(modifier_token, initializer_token)

        params_to_freeze = itertools.chain(
            self.vae.parameters(),
            self.text_encoder.text_model.encoder.parameters(),
            self.text_encoder.text_model.final_layer_norm.parameters(),
            self.text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        for param in params_to_freeze:
            param.requires_grad = False

        self.prior_loss_weight = prior_loss_weight

    def get_params_to_optimize(self):
        # the itertools.chain only support one pass iteration
        params_to_optimize = itertools.chain(
            self.text_encoder.get_input_embeddings().parameters(),
            [x[1] for x in self.unet.named_parameters() if ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0])])
        return params_to_optimize

    def init_new_concept(self, modifier_token, initializer_token):
        '''

        Args:
            modifier_token: <new1>+<new2>+...
            initializer_token: rare world 1+rare world 2+..., use those embedding to initialize modifier tokens

        Returns:
            modifier_token: [<new1>, <new2>, ...]
            modifier_token_id: [id1, id2]

        '''

        modifier_token_id = []
        initializer_token_id = []

        modifier_token = modifier_token.split('+')
        initializer_token = initializer_token.split('+')
        if len(modifier_token) > len(initializer_token):
            raise ValueError('You must specify + separated initializer token for each modifier token.')
        for mod_token, init_token in zip(modifier_token, initializer_token[:len(modifier_token)]):
            # Add the placeholder token in tokenizer
            num_added_tokens = self.tokenizer.add_tokens(mod_token)
            if num_added_tokens == 0:
                raise ValueError(f'The tokenizer already contains the token {mod_token}. Please pass a different'
                                 ' `modifier_token` that is not already in the tokenizer.')

            # Convert the initializer_token, placeholder_token to ids
            token_ids = self.tokenizer.encode([init_token], add_special_tokens=False)
            # print(token_ids)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError('The initializer token must be a single token.')

            initializer_token_id.append(token_ids[0])
            modifier_token_id.append(self.tokenizer.convert_tokens_to_ids(mod_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        for (x, y) in zip(modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]
        return modifier_token, modifier_token_id

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
        encoder_hidden_states = self.text_encoder(text_input_ids)[0]

        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.scheduler.config.prediction_type == 'v_prediction':
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f'Unknown prediction type {self.scheduler.config.prediction_type}')

        if self.prior_loss_weight is not None:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            mask = torch.chunk(masks, 2, dim=0)[0]
            # Compute instance loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction='none')
            loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction='mean')

            # Add the prior loss to the instance loss.
            loss = loss + self.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction='none')
            loss = ((loss * masks).sum([1, 2, 3]) / masks.sum([1, 2, 3])).mean()
        return loss

    def load_delta_state_dict(self, delta_state_dict):
        # load text encoder
        logger = get_root_logger()

        if 'text_encoder' in delta_state_dict:
            logger.info('loading text encoder')
            self.text_encoder.load_state_dict(delta_state_dict['text_encoder'])

        if 'new_concept_embedding' in delta_state_dict and len(delta_state_dict['new_concept_embedding']) != 0:
            new_concept_token = list(delta_state_dict['new_concept_embedding'].keys())

            # check whether new concept is initialized,
            if len(new_concept_token) != len(self.modifier_token):
                logger.warning('Your checkpoint have different concept with your model, loading existing concepts')

            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            for i, id_ in enumerate(self.modifier_token_id):
                logger.info(f'load: token_{id_} from {self.modifier_token[i]}')
                token_embeds[id_] = delta_state_dict['new_concept_embedding'][self.modifier_token[i]]

        # load new token
        if 'modifier_token' in delta_state_dict:
            modifier_tokens = list(delta_state_dict['modifier_token'].keys())

            # check whether new concept is initialized,
            if len(modifier_tokens) == len(self.modifier_token) and set(modifier_tokens) == set(self.modifier_token):
                token_embeds = self.text_encoder.get_input_embeddings().weight.data
                for i, id_ in enumerate(self.modifier_token_id):
                    logger.info(f'load: token_{id_} from {self.modifier_token[i]}')
                    token_embeds[id_] = delta_state_dict['modifier_token'][self.modifier_token[i]]
            else:
                raise Exception('Your checkpoint have different concept with your model, please check')

        # load attention
        for name, params in self.unet.named_parameters():
            if self.finetune_params == 'crossattn':
                if 'attn2' in name:
                    params.data.copy_(delta_state_dict['unet'][f'{name}'])
            elif self.finetune_params == 'crossattn_kv':
                if 'attn2.to_k' in name or 'attn2.to_v' in name:
                    logger.info(f'loading: {name}')
                    params.data.copy_(delta_state_dict['unet'][f'{name}'])
            else:
                raise NotImplementedError

    def delta_state_dict(self):
        delta_dict = {'unet': {}, 'modifier_token': {}}
        if self.modifier_token is not None:
            for i in range(len(self.modifier_token_id)):
                learned_embeds = self.text_encoder.get_input_embeddings().weight[self.modifier_token_id[i]]
                delta_dict['modifier_token'][self.modifier_token[i]] = learned_embeds.detach().cpu()

        for name, params in self.unet.named_parameters():
            if self.finetune_params == 'crossattn':
                if 'attn2' in name:
                    delta_dict['unet'][name] = params.cpu().clone()
            elif self.finetune_params == 'crossattn_kv':
                if 'attn2.to_k' in name or 'attn2.to_v' in name:
                    delta_dict['unet'][name] = params.cpu().clone()
            else:
                raise NotImplementedError
        return delta_dict

    def create_custom_diffusion(self):
        #
        for name, params in self.unet.named_parameters():
            if self.finetune_params == 'crossattn':
                if 'attn2' in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            elif self.finetune_params == 'crossattn_kv':
                if 'attn2.to_k' in name or 'attn2.to_v' in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                raise NotImplementedError

        def new_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
                crossattn = False
            else:
                crossattn = True

            if self.cross_attention_norm:
                encoder_hidden_states = self.norm_cross(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            if crossattn:
                modifier = torch.ones_like(key)
                modifier[:, :1, :] = modifier[:, :1, :] * 0.
                key = modifier * key + (1 - modifier) * key.detach()
                value = modifier * value + (1 - modifier) * value.detach()

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            return hidden_states

        def change_forward(unet):
            for layer in unet.children():
                if type(layer) == CrossAttention:
                    bound_method = new_forward.__get__(layer, layer.__class__)
                    setattr(layer, 'forward', bound_method)
                else:
                    change_forward(layer)

        change_forward(self.unet)
