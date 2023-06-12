import itertools
import torch
import torch.nn.functional as F

from mixofshow.archs.stable_diffusion_arch import Stable_Diffusion
from mixofshow.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class Dreambooth(Stable_Diffusion):

    def __init__(self,
                 pretrained_path,
                 train_text_encoder=False,
                 sd_version='v1',
                 test_sampler_type='ddim',
                 prior_loss_weight=None):
        super().__init__(pretrained_path=pretrained_path, sd_version=sd_version, test_sampler_type=test_sampler_type)

        self.train_text_encoder = train_text_encoder
        self.prior_loss_weight = prior_loss_weight

        # freeze params
        params_to_freeze = itertools.chain(self.vae.parameters(),
                                           self.text_encoder.parameters() if not self.train_text_encoder else [])
        for param in params_to_freeze:
            param.requires_grad = False

    def get_params_to_optimize(self):
        params_to_optimize = itertools.chain(self.unet.parameters(), self.text_encoder.parameters()) \
            if self.train_text_encoder else self.unet.parameters()
        return params_to_optimize

    def forward(self, images, prompts, masks=None):
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

            # Compute instance loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')

            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction='mean')

            # Add the prior loss to the instance loss.
            loss = loss + self.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')
        return loss

    def load_delta_state_dict(self, delta_state_dict):
        # load text encoder
        if 'text_encoder' in delta_state_dict:
            self.text_encoder.load_state_dict(delta_state_dict['text_encoder'])

        self.unet.load_state_dict(delta_state_dict['unet'])

    def delta_state_dict(self):
        delta_dict = {'unet': self.unet.state_dict()}
        if self.train_text_encoder:
            delta_dict['text_encoder'] = self.text_encoder.state_dict()
        return delta_dict
