import argparse
import copy
import os
import os.path as osp

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from omegaconf import OmegaConf

from mixofshow.data.lora_dataset import LoraDataset
from mixofshow.data.prompt_dataset import PromptDataset
from mixofshow.pipelines.pipeline_edlora import EDLoRAPipeline, StableDiffusionPipeline
from mixofshow.pipelines.trainer_edlora import EDLoRATrainer
from mixofshow.utils.convert_edlora_to_diffusers import convert_edlora
from mixofshow.utils.util import MessageLogger, dict2str, reduce_loss_dict, set_path_logger
from test_edlora import visual_validation

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version('0.18.2')


def train(root_path, args):

    # load config
    opt = OmegaConf.to_container(OmegaConf.load(args.opt), resolve=True)

    # set accelerator, mix-precision set in the environment by "accelerate config"
    accelerator = Accelerator(mixed_precision=opt['mixed_precision'], gradient_accumulation_steps=opt['gradient_accumulation_steps'])

    # set experiment dir
    with accelerator.main_process_first():
        set_path_logger(accelerator, root_path, args.opt, opt, is_train=True)

    # get logger
    logger = get_logger('mixofshow', log_level='INFO')
    logger.info(accelerator.state, main_process_only=True)

    logger.info(dict2str(opt))

    # If passed along, set the training seed now.
    if opt.get('manual_seed') is not None:
        set_seed(opt['manual_seed'])

    # Load model
    EDLoRA_trainer = EDLoRATrainer(**opt['models'])

    # set optimizer
    train_opt = opt['train']
    optim_type = train_opt['optim_g'].pop('type')
    assert optim_type == 'AdamW', 'only support AdamW now'
    optimizer = torch.optim.AdamW(EDLoRA_trainer.get_params_to_optimize(), **train_opt['optim_g'])

    # Get the training dataset
    trainset_cfg = opt['datasets']['train']
    train_dataset = LoraDataset(trainset_cfg)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=trainset_cfg['batch_size_per_gpu'], shuffle=True, drop_last=True)

    # Get the training dataset
    valset_cfg = opt['datasets']['val_vis']
    val_dataset = PromptDataset(valset_cfg)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=valset_cfg['batch_size_per_gpu'], shuffle=False)

    # Prepare everything with our `accelerator`.
    EDLoRA_trainer, optimizer, train_dataloader, val_dataloader = accelerator.prepare(EDLoRA_trainer, optimizer, train_dataloader, val_dataloader)

    # Train!
    total_batch_size = opt['datasets']['train']['batch_size_per_gpu'] * accelerator.num_processes * opt['gradient_accumulation_steps']
    total_iter = len(train_dataset) / total_batch_size
    opt['train']['total_iter'] = total_iter

    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f"  Instantaneous batch size per device = {opt['datasets']['train']['batch_size_per_gpu']}")
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Total optimization steps = {total_iter}')
    global_step = 0

    # Scheduler
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_iter * opt['gradient_accumulation_steps'],
    )

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)

    msg_logger = MessageLogger(opt, global_step)
    stop_emb_update = False

    original_embedding = copy.deepcopy(accelerator.unwrap_model(EDLoRA_trainer).text_encoder.get_input_embeddings().weight)

    while global_step < opt['train']['total_iter']:
        with accelerator.accumulate(EDLoRA_trainer):

            accelerator.unwrap_model(EDLoRA_trainer).unet.train()
            accelerator.unwrap_model(EDLoRA_trainer).text_encoder.train()
            loss_dict = {}

            batch = next(train_data_yielder)

            if 'masks' in batch:
                masks = batch['masks']
            else:
                masks = batch['img_masks']

            loss = EDLoRA_trainer(batch['images'], batch['prompts'], masks, batch['img_masks'])
            loss_dict['loss'] = loss

            # get fix embedding and learn embedding
            index_no_updates = torch.arange(len(accelerator.unwrap_model(EDLoRA_trainer).tokenizer)) != -1
            if not stop_emb_update:
                for token_id in accelerator.unwrap_model(EDLoRA_trainer).get_all_concept_token_ids():
                    index_no_updates[token_id] = False

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            # set no update token to origin
            token_embeds = accelerator.unwrap_model(EDLoRA_trainer).text_encoder.get_input_embeddings().weight
            token_embeds.data[index_no_updates, :] = original_embedding.data[index_no_updates, :]

            token_embeds = accelerator.unwrap_model(EDLoRA_trainer).text_encoder.get_input_embeddings().weight
            concept_token_ids = accelerator.unwrap_model(EDLoRA_trainer).get_all_concept_token_ids()
            loss_dict['Norm_mean'] = token_embeds[concept_token_ids].norm(dim=-1).mean()
            if stop_emb_update is False and float(loss_dict['Norm_mean']) >= train_opt.get('emb_norm_threshold', 5.5e-1):
                stop_emb_update = True
                original_embedding = copy.deepcopy(accelerator.unwrap_model(EDLoRA_trainer).text_encoder.get_input_embeddings().weight)

            log_dict = reduce_loss_dict(accelerator, loss_dict)

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            global_step += 1

            if global_step % opt['logger']['print_freq'] == 0:
                log_vars = {'iter': global_step}
                log_vars.update({'lrs': lr_scheduler.get_last_lr()})
                log_vars.update(log_dict)
                msg_logger(log_vars)

            if global_step % opt['logger']['save_checkpoint_freq'] == 0:
                save_and_validation(accelerator, opt, EDLoRA_trainer, val_dataloader, global_step, logger)

    # Save the lora layers, final eval
    accelerator.wait_for_everyone()
    save_and_validation(accelerator, opt, EDLoRA_trainer, val_dataloader, 'latest', logger)


def save_and_validation(accelerator, opt, EDLoRA_trainer, val_dataloader, global_step, logger):
    enable_edlora = opt['models']['enable_edlora']
    lora_type = 'edlora' if enable_edlora else 'lora'
    save_path = os.path.join(opt['path']['models'], f'{lora_type}_model-{global_step}.pth')

    if accelerator.is_main_process:
        accelerator.save({'params': accelerator.unwrap_model(EDLoRA_trainer).delta_state_dict()}, save_path)
        logger.info(f'Save state to {save_path}')

    accelerator.wait_for_everyone()

    if opt['val']['val_during_save']:
        logger.info(f'Start validation {save_path}:')
        for lora_alpha in opt['val']['alpha_list']:
            pipeclass = EDLoRAPipeline if enable_edlora else StableDiffusionPipeline

            pipe = pipeclass.from_pretrained(opt['models']['pretrained_path'],
                scheduler=DPMSolverMultistepScheduler.from_pretrained(opt['models']['pretrained_path'], subfolder='scheduler'),
                torch_dtype=torch.float16).to('cuda')
            pipe, new_concept_cfg = convert_edlora(pipe, torch.load(save_path), enable_edlora=enable_edlora, alpha=lora_alpha)
            pipe.set_new_concept_cfg(new_concept_cfg)
            pipe.set_progress_bar_config(disable=True)
            visual_validation(accelerator, pipe, val_dataloader, f'Iters-{global_step}_Alpha-{lora_alpha}', opt)

            del pipe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/train/EDLoRA/EDLoRA_hina_Anyv4_B4_Iter1K.yml')
    args = parser.parse_args()

    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train(root_path, args)
