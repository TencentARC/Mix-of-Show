import argparse
import os
import os.path as osp

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import check_min_version
from omegaconf import OmegaConf
from tqdm import tqdm

from mixofshow.data.prompt_dataset import PromptDataset
from mixofshow.pipelines.pipeline_edlora import EDLoRAPipeline, StableDiffusionPipeline
from mixofshow.utils.convert_edlora_to_diffusers import convert_edlora
from mixofshow.utils.util import NEGATIVE_PROMPT, compose_visualize, dict2str, pil_imwrite, set_path_logger

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version('0.18.2')


def visual_validation(accelerator, pipe, dataloader, current_iter, opt):
    dataset_name = dataloader.dataset.opt['name']
    pipe.unet.eval()
    pipe.text_encoder.eval()

    for idx, val_data in enumerate(tqdm(dataloader)):
        output = pipe(
            prompt=val_data['prompts'],
            latents=val_data['latents'].to(dtype=torch.float16),
            negative_prompt=[NEGATIVE_PROMPT] * len(val_data['prompts']),
            num_inference_steps=opt['val']['sample'].get('num_inference_steps', 50),
            guidance_scale=opt['val']['sample'].get('guidance_scale', 7.5),
        ).images

        for img, prompt, indice in zip(output, val_data['prompts'], val_data['indices']):
            img_name = '{prompt}---G_{guidance_scale}_S_{steps}---{indice}'.format(
                prompt=prompt.replace(' ', '_'),
                guidance_scale=opt['val']['sample'].get('guidance_scale', 7.5),
                steps=opt['val']['sample'].get('num_inference_steps', 50),
                indice=indice)

            save_img_path = osp.join(opt['path']['visualization'], dataset_name, f'{current_iter}', f'{img_name}---{current_iter}.png')

            pil_imwrite(img, save_img_path)
        # tentative for out of GPU memory
        del output
        torch.cuda.empty_cache()

    # Save the lora layers, final eval
    accelerator.wait_for_everyone()

    if opt['val'].get('compose_visualize'):
        if accelerator.is_main_process:
            compose_visualize(os.path.dirname(save_img_path))


def test(root_path, args):

    # load config
    opt = OmegaConf.to_container(OmegaConf.load(args.opt), resolve=True)

    # set accelerator, mix-precision set in the environment by "accelerate config"
    accelerator = Accelerator(mixed_precision=opt['mixed_precision'])

    # set experiment dir
    with accelerator.main_process_first():
        set_path_logger(accelerator, root_path, args.opt, opt, is_train=False)

    # get logger
    logger = get_logger('mixofshow', log_level='INFO')
    logger.info(accelerator.state, main_process_only=True)

    logger.info(dict2str(opt))

    # If passed along, set the training seed now.
    if opt.get('manual_seed') is not None:
        set_seed(opt['manual_seed'])

    # Get the training dataset
    valset_cfg = opt['datasets']['val_vis']
    val_dataset = PromptDataset(valset_cfg)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=valset_cfg['batch_size_per_gpu'], shuffle=False)

    enable_edlora = opt['models']['enable_edlora']

    for lora_alpha in opt['val']['alpha_list']:
        pipeclass = EDLoRAPipeline if enable_edlora else StableDiffusionPipeline
        pipe = pipeclass.from_pretrained(opt['models']['pretrained_path'],
            scheduler=DPMSolverMultistepScheduler.from_pretrained(opt['models']['pretrained_path'], subfolder='scheduler'),
            torch_dtype=torch.float16).to('cuda')
        pipe, new_concept_cfg = convert_edlora(pipe, torch.load(opt['path']['lora_path']), enable_edlora=enable_edlora, alpha=lora_alpha)
        pipe.set_new_concept_cfg(new_concept_cfg)
        # visualize embedding + LoRA weight shift
        logger.info(f'Start validation sample lora({lora_alpha}):')

        lora_type = 'edlora' if enable_edlora else 'lora'
        visual_validation(accelerator, pipe, val_dataloader, f'validation_{lora_type}_{lora_alpha}', opt)
        del pipe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/test/EDLoRA/EDLoRA_hina_Anyv4_B4_Iter1K.yml')
    args = parser.parse_args()

    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test(root_path, args)
