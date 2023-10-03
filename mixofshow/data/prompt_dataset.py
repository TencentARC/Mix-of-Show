import os
import random
import re

import torch
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    'A simple dataset to prepare the prompts to generate class images on multiple GPUs.'

    def __init__(self, opt):
        self.opt = opt

        self.prompts = opt['prompts']

        if isinstance(self.prompts, list):
            self.prompts = self.prompts
        elif os.path.exists(self.prompts):
            # is file
            with open(self.prompts, 'r') as fr:
                lines = fr.readlines()
                lines = [item.strip() for item in lines]
            self.prompts = lines
        else:
            raise ValueError(
                'prompts should be a prompt file path or prompt list, please check!'
            )

        self.prompts = self.replace_placeholder(self.prompts)

        self.num_samples_per_prompt = opt['num_samples_per_prompt']
        self.prompts_to_generate = [
            (p, i) for i in range(1, self.num_samples_per_prompt + 1)
            for p in self.prompts
        ]
        self.latent_size = opt['latent_size']  # (4,64,64)
        self.share_latent_across_prompt = opt.get('share_latent_across_prompt', True)  # (true, false)

    def replace_placeholder(self, prompts):
        # replace placehold token
        replace_mapping = self.opt.get('replace_mapping', {})
        new_lines = []
        for line in self.prompts:
            if len(line.strip()) == 0:
                continue
            for k, v in replace_mapping.items():
                line = line.replace(k, v)
            line = line.strip()
            line = re.sub(' +', ' ', line)
            new_lines.append(line)
        return new_lines

    def __len__(self):
        return len(self.prompts_to_generate)

    def __getitem__(self, index):
        prompt, indice = self.prompts_to_generate[index]
        example = {}
        example['prompts'] = prompt
        example['indices'] = indice
        if self.share_latent_across_prompt:
            seed = indice
        else:
            seed = random.randint(0, 1000)
        example['latents'] = torch.randn(self.latent_size, generator=torch.manual_seed(seed))
        return example
