import torch
from collections import OrderedDict
from torch.nn.utils import clip_grad_norm_

from mixofshow.data.prompt_dataset import PromptDataset
from mixofshow.utils.registry import MODEL_REGISTRY
from .finetune_model import FinetuneModel


@MODEL_REGISTRY.register()
class CustomDiffusionModel(FinetuneModel):

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()
        self.get_bare_model(self.net_g).unet.train()
        self.get_bare_model(self.net_g).text_encoder.train()

        self.optimizer_g.zero_grad()
        loss = self.net_g(self.images, self.prompts, self.masks)
        loss_dict['loss'] = loss
        loss.backward()

        # Zero out the gradients for all token embeddings except the newly added
        # embeddings for the concept, as we only want to optimize the concept embeddings
        grads_text_encoder = self.get_bare_model(self.net_g).text_encoder.get_input_embeddings().weight.grad
        # Get the index for tokens that we want to zero the grads for
        index_grads_to_zero = torch.arange(len(self.get_bare_model(self.net_g).tokenizer)) != self.get_bare_model(
            self.net_g).modifier_token_id[0]
        for index in self.get_bare_model(self.net_g).modifier_token_id[1:]:
            index_grads_to_zero = index_grads_to_zero & (
                torch.arange(len(self.get_bare_model(self.net_g).tokenizer)) != index)
        assert set(torch.where(index_grads_to_zero == False)[0].tolist()) == set(  # noqa: E712
            self.get_bare_model(self.net_g).modifier_token_id), 'error grad mask'

        grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[index_grads_to_zero, :].fill_(0)

        if self.opt['train'].get('max_grad_norm'):
            clip_grad_norm_(self.net_g.parameters(), max_norm=self.opt['train']['max_grad_norm'])

        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if isinstance(dataloader.dataset, (PromptDataset)):
            # sample negprompt
            self.opt['val']['sample']['use_negative_prompt'] = True
            self.visual_validation(dataloader, f'{current_iter}_negprompt', tb_logger, save_img)
        else:
            self.loss_validation(dataloader, current_iter, tb_logger)
