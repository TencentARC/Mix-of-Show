import json
import numpy as np
import os
import random
import re
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from mixofshow.data.pil_transform import PairCompose, build_transform
from mixofshow.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class LoraDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, opt):
        self.opt = opt
        self.instance_images_path = []

        with open(opt['concept_list'], 'r') as f:
            concept_list = json.load(f)

        replace_mapping = opt.get('replace_mapping', {})
        use_caption = opt.get('use_caption', False)

        for concept in concept_list:
            instance_prompt = concept['instance_prompt']
            caption_dir = concept.get('caption_dir')

            instance_prompt = self.process_text(instance_prompt, replace_mapping)

            inst_img_path = []
            for x in Path(concept['instance_data_dir']).iterdir():
                if x.is_file() and x.name != '.DS_Store':
                    basename = os.path.splitext(os.path.basename(x))[0]
                    caption_path = os.path.join(caption_dir, f'{basename}.txt') if caption_dir is not None else None
                    if use_caption and caption_path is not None and os.path.exists(caption_path):
                        with open(caption_path, 'r') as fr:
                            line = fr.readlines()[0]
                            instance_prompt_image = self.process_text(line, replace_mapping)
                    else:
                        instance_prompt_image = instance_prompt
                    inst_img_path.append((x, instance_prompt_image))

            self.instance_images_path.extend(inst_img_path)

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)

        self.instance_transform = PairCompose(
            [build_transform(transform_opt) for transform_opt in opt['instance_transform']])

    def process_text(self, instance_prompt, replace_mapping):
        for k, v in replace_mapping.items():
            instance_prompt = instance_prompt.replace(k, v)
        instance_prompt = instance_prompt.strip()
        instance_prompt = re.sub(' +', ' ', instance_prompt)
        return instance_prompt

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image).convert('RGB')

        extra_args = {'prompts': instance_prompt}

        instance_image, extra_args = self.instance_transform(instance_image, **extra_args)
        example['images'] = instance_image
        if 'mask' in extra_args:
            example['masks'] = extra_args['mask']
        else:
            size_h, size_w = example['images'].shape[1:]
            mask = np.ones((size_h // 8, size_w // 8))
            example['masks'] = torch.from_numpy(mask).unsqueeze(0)
        example['prompts'] = extra_args['prompts']
        example['masks'] = example['masks'].unsqueeze(0)
        return example
