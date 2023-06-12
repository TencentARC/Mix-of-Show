import json
import numpy as np
import os
import PIL
import random
import torch
from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from mixofshow.utils.registry import DATASET_REGISTRY

ImageFile.LOAD_TRUNCATED_IMAGES = True


@DATASET_REGISTRY.register()
class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, opt):
        self.opt = opt
        self.size = opt['size']
        self.center_crop = opt['center_crop']
        self.interpolation = PIL.Image.BILINEAR

        self.instance_images_path = []
        self.class_images_path = []
        self.num_class_images = opt['num_class_images']
        self.with_prior_preservation = opt['with_prior_preservation']

        with open(opt['concept_list'], 'r') as f:
            concept_list = json.load(f)

        for concept in concept_list:
            instance_prompt = concept['instance_prompt']
            instance_prompt = instance_prompt.strip()
            inst_img_path = [(x, instance_prompt) for x in Path(concept['instance_data_dir']).iterdir() if x.is_file()]
            self.instance_images_path.extend(inst_img_path)

            if self.with_prior_preservation:
                class_data_root = Path(concept['class_data_dir'])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept['class_prompt'] for _ in range(len(class_images_path))]
                else:
                    with open(class_data_root, 'r') as f:
                        class_images_path = f.read().splitlines()
                    with open(concept['class_prompt'], 'r') as f:
                        class_prompt = f.read().splitlines()

                class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
                self.class_images_path.extend(class_img_path[:self.num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * opt['hflip'])

        self.image_transforms = transforms.Compose([
            self.flip,
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size) if self.center_crop else transforms.RandomCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == 'RGB':
            instance_image = instance_image.convert('RGB')
        instance_image = self.flip(instance_image)

        # apply augmentation and create a valid image regions mask #
        if np.random.randint(0, 3) < 2:
            random_scale = np.random.randint(self.size // 3, self.size + 1)
        else:
            random_scale = np.random.randint(int(1.2 * self.size), int(1.4 * self.size))

        if random_scale % 2 == 1:
            random_scale += 1

        if random_scale < 0.6 * self.size:
            add_to_caption = np.random.choice(['a far away ', 'very small '])
            instance_prompt = add_to_caption + instance_prompt
            cx = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
            cy = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)

            instance_image1 = instance_image.resize((random_scale, random_scale), resample=self.interpolation)
            instance_image1 = np.array(instance_image1).astype(np.uint8)
            instance_image1 = (instance_image1 / 127.5 - 1.0).astype(np.float32)

            instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
            instance_image[cx - random_scale // 2:cx + random_scale // 2,
                           cy - random_scale // 2:cy + random_scale // 2, :] = instance_image1

            mask = np.zeros((self.size // 8, self.size // 8))
            mask[(cx - random_scale // 2) // 8 + 1:(cx + random_scale // 2) // 8 - 1,
                 (cy - random_scale // 2) // 8 + 1:(cy + random_scale // 2) // 8 - 1] = 1.
        elif random_scale > self.size:
            add_to_caption = np.random.choice(['zoomed in ', 'close up '])
            instance_prompt = add_to_caption + instance_prompt
            cx = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)
            cy = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)

            instance_image = instance_image.resize((random_scale, random_scale), resample=self.interpolation)
            instance_image = np.array(instance_image).astype(np.uint8)
            instance_image = (instance_image / 127.5 - 1.0).astype(np.float32)
            instance_image = instance_image[cx - self.size // 2:cx + self.size // 2,
                                            cy - self.size // 2:cy + self.size // 2, :]
            mask = np.ones((self.size // 8, self.size // 8))
        else:
            if self.size is not None:
                instance_image = instance_image.resize((self.size, self.size), resample=self.interpolation)
            instance_image = np.array(instance_image).astype(np.uint8)
            instance_image = (instance_image / 127.5 - 1.0).astype(np.float32)
            mask = np.ones((self.size // 8, self.size // 8))
        ########################################################################

        example['instance_image'] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example['instance_mask'] = torch.from_numpy(mask)
        example['instance_prompt'] = instance_prompt

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == 'RGB':
                class_image = class_image.convert('RGB')
            example['class_image'] = self.image_transforms(class_image)
            mask = np.ones((self.size // 8, self.size // 8))
            example['class_mask'] = torch.from_numpy(mask)
            example['class_prompt'] = class_prompt
        return example
