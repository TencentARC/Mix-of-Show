from collections import OrderedDict
from torch.nn.utils import clip_grad_norm_

from mixofshow.data.prompt_dataset import PromptDataset
from mixofshow.utils.registry import MODEL_REGISTRY
from .finetune_model import FinetuneModel


@MODEL_REGISTRY.register()
class DreamBoothModel(FinetuneModel):

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()
        self.get_bare_model(self.net_g).unet.train()
        if self.get_bare_model(self.net_g).train_text_encoder:
            self.get_bare_model(self.net_g).text_encoder.train()

        self.optimizer_g.zero_grad()
        loss = self.net_g(self.images, self.prompts)
        loss_dict['loss'] = loss
        loss.backward()

        if self.opt['train'].get('max_grad_norm'):
            clip_grad_norm_(self.net_g.parameters(), max_norm=self.opt['train']['max_grad_norm'])

        # todo: we omit the clip grad norm here
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if isinstance(dataloader.dataset, (PromptDataset)):
            # sample negprompt
            self.opt['val']['sample']['use_negative_prompt'] = True
            self.visual_validation(dataloader, f'{current_iter}_negprompt', tb_logger, save_img)
        else:
            self.loss_validation(dataloader, current_iter, tb_logger)
