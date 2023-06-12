import os
import os.path
import time
import torch
import torch.distributed as dist
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from mixofshow.archs import build_network
from mixofshow.data.prompt_dataset import PromptDataset
from mixofshow.utils import get_root_logger
from mixofshow.utils.dist_util import master_only
from mixofshow.utils.img_util import compose_visualize, pil_imwrite
from mixofshow.utils.misc import AverageMeter
from mixofshow.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class FinetuneModel(BaseModel):

    def __init__(self, opt):
        super(FinetuneModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_delta_network(self.net_g, load_path, param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        # define losses
        # within model

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # optimizer g
        lr = train_opt['optim_g']['lr']
        scale_lr = train_opt['optim_g'].pop('scale_lr')
        if scale_lr:
            if dist.is_initialized():
                lr = lr * dist.get_world_size() * self.opt['datasets']['train']['batch_size_per_gpu']
            else:
                lr = lr * self.opt['datasets']['train']['batch_size_per_gpu']
        train_opt['optim_g']['lr'] = lr

        logger = get_root_logger()
        logger.info('Scale learning rate to: %.2e' % train_opt['optim_g']['lr'])

        optim_type = train_opt['optim_g'].pop('type')

        self.optimizer_g = self.get_optimizer(optim_type,
                                              self.get_bare_model(self.net_g).get_params_to_optimize(),
                                              **train_opt['optim_g'])
        self.optimizer_g.zero_grad()
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.prompts = data['prompts']
        self.images = data['images'].to(self.device) if 'images' in data else None
        self.masks = data['masks'].to(self.device) if 'masks' in data else None
        self.latents = data['latents'].to(self.device) if 'latents' in data else None

    def save(self, epoch, current_iter):
        self.save_delta_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    @torch.no_grad()
    def test(self, test_loss=True):
        if test_loss is True:
            self.get_bare_model(self.net_g).unet.eval()
            self.get_bare_model(self.net_g).text_encoder.eval()
            loss = self.net_g(self.images, self.prompts, self.masks)
            return loss
        else:
            self.get_bare_model(self.net_g).unet.eval()
            self.get_bare_model(self.net_g).text_encoder.eval()
            self.output = self.get_bare_model(self.net_g).sample(
                self.prompts,
                self.latents,
                use_negative_prompt=self.opt['val']['sample'].get('use_negative_prompt', False),
                num_inference_steps=self.opt['val']['sample'].get('num_inference_steps', 50),
                guidance_scale=self.opt['val']['sample'].get('guidance_scale', 7.5))

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # first stage
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if isinstance(dataloader.dataset, (PromptDataset)):
            self.visual_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            self.loss_validation(dataloader, current_iter, tb_logger)

    def loss_validation(self, dataloader, current_iter, tb_logger):
        loss_meter = AverageMeter()
        use_pbar = self.opt['val'].get('pbar', False)

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            loss = self.test(test_loss=True)
            loss_meter.update(loss.item(), len(val_data['prompts']))
            if use_pbar:
                pbar.update(1)

        if use_pbar:
            pbar.close()

        if dist.is_initialized():
            dist.barrier()
        logger = get_root_logger()
        logger.info('Val Loss: %.4f' % loss_meter.avg)
        if tb_logger and self.opt['rank'] == 0:
            tb_logger.add_scalar('metrics/Val Loss', loss_meter.avg, current_iter)

    def visual_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        use_pbar = self.opt['val'].get('pbar', False)

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test(test_loss=False)
            visuals = self.get_current_visuals()  # image list
            for img, prompt, indice in zip(visuals['results'], val_data['prompts'], val_data['indices']):
                img_name = '{prompt}---G_{guidance_scale}_S_{steps}---{indice}'.format(
                    prompt=prompt.replace(' ', '_'),
                    guidance_scale=self.opt['val']['sample'].get('guidance_scale', 7.5),
                    steps=self.opt['val']['sample'].get('num_inference_steps', 50),
                    indice=indice)
                if save_img:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'iters_{current_iter}', f'{img_name}---{current_iter}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}---{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'iters_{current_iter}', f'{img_name}---{current_iter}.png')
                    pil_imwrite(img, save_img_path)
            # tentative for out of GPU memory
            del self.output
            torch.cuda.empty_cache()

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if dist.is_initialized():
            dist.barrier()

        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.eval_performance(dataloader, current_iter, tb_logger)

        if save_img and self.opt['val'].get('compose_visualize'):
            compose_visualize(os.path.dirname(save_img_path))

    @master_only
    def eval_performance(self, dataloader, current_iter, tb_logger):
        raise NotImplementedError

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['results'] = self.output
        return out_dict

    def load_delta_network(self, net, load_path, param_key='params'):
        """Load delta network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        net.load_delta_state_dict(load_net)

    @master_only
    def save_delta_network(self, net, net_label, current_iter, param_key='params'):
        """Save delta networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.delta_state_dict()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')
