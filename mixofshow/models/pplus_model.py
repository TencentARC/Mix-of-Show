import torch
from collections import OrderedDict
from torch.nn.utils import clip_grad_norm_

from mixofshow.data.prompt_dataset import PromptDataset
from mixofshow.losses import build_loss
from mixofshow.utils.registry import MODEL_REGISTRY
from .finetune_model import FinetuneModel


@MODEL_REGISTRY.register()
class PPlusModel(FinetuneModel):

    def init_training_settings(self):
        # define losses

        if self.opt['train'].get('kde_opt'):
            self.kde_loss = build_loss(self.opt['train']['kde_opt']).to(self.device)
        else:
            self.kde_loss = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        self.get_bare_model(self.net_g).unet.train()
        self.get_bare_model(self.net_g).text_encoder.train()

        self.optimizer_g.zero_grad()
        loss = self.net_g(self.images, self.prompts, self.masks)
        loss_dict['loss'] = loss

        # get fix embedding and learn embedding
        index_no_updates = torch.arange(len(self.get_bare_model(self.net_g).tokenizer)) != -1
        for token_id in self.get_bare_model(self.net_g).get_all_concept_token_ids():
            index_no_updates[token_id] = False

        if self.kde_loss:
            fix_embedding = self.get_bare_model(self.net_g).text_encoder.get_input_embeddings().weight[index_no_updates]
            learn_embedding = self.get_bare_model(
                self.net_g).text_encoder.get_input_embeddings().weight[~index_no_updates]
            token_reg = self.kde_loss(fix_embedding, learn_embedding)
            loss_dict['token_reg'] = token_reg
            loss += token_reg

        loss.backward()

        grads_text_encoder = self.get_bare_model(self.net_g).text_encoder.get_input_embeddings().weight.grad
        if grads_text_encoder is not None:
            grads_text_encoder.data[index_no_updates, :] = grads_text_encoder.data[index_no_updates, :].fill_(0)

        if self.opt['train'].get('max_grad_norm'):
            clip_grad_norm_(self.net_g.parameters(), max_norm=self.opt['train']['max_grad_norm'])

        self.optimizer_g.step()

        token_embeds = self.get_bare_model(self.net_g).text_encoder.get_input_embeddings().weight
        concept_token_ids = self.get_bare_model(self.net_g).get_all_concept_token_ids()
        loss_dict['Norm_mean'] = token_embeds[concept_token_ids].norm(dim=-1).mean()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if isinstance(dataloader.dataset, (PromptDataset)):
            # sample negprompt
            self.opt['val']['sample']['use_negative_prompt'] = True
            self.visual_validation(dataloader, f'{current_iter}_negprompt', tb_logger, save_img)
        else:
            self.loss_validation(dataloader, current_iter, tb_logger)
