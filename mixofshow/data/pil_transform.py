import inspect
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import CenterCrop, Normalize, RandomCrop, RandomHorizontalFlip, Resize
from torchvision.transforms.functional import InterpolationMode

from mixofshow.utils.registry import TRANSFORM_REGISTRY


def build_transform(opt):
    """Build performance evaluator from options.
    Args:
        opt (dict): Configuration.
    """
    opt = deepcopy(opt)
    transform_type = opt.pop('type')
    transform = TRANSFORM_REGISTRY.get(transform_type)(**opt)
    return transform


TRANSFORM_REGISTRY.register(Normalize)
TRANSFORM_REGISTRY.register(Resize)
TRANSFORM_REGISTRY.register(RandomHorizontalFlip)
TRANSFORM_REGISTRY.register(CenterCrop)
TRANSFORM_REGISTRY.register(RandomCrop)


@TRANSFORM_REGISTRY.register()
class BILINEARResize(Resize):
    def __init__(self, size):
        super(BILINEARResize,
              self).__init__(size, interpolation=InterpolationMode.BILINEAR)


@TRANSFORM_REGISTRY.register()
class PairRandomCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            self.height, self.width = size, size
        else:
            self.height, self.width = size

    def forward(self, img, **kwargs):
        img_width, img_height = img.size
        mask_width, mask_height = kwargs['mask'].size

        assert img_height >= self.height and img_height == mask_height
        assert img_width >= self.width and img_width == mask_width

        x = random.randint(0, img_width - self.width)
        y = random.randint(0, img_height - self.height)
        img = F.crop(img, y, x, self.height, self.width)
        kwargs['mask'] = F.crop(kwargs['mask'], y, x, self.height, self.width)
        return img, kwargs


@TRANSFORM_REGISTRY.register()
class ToTensor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pic):
        return F.to_tensor(pic)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


@TRANSFORM_REGISTRY.register()
class PairRandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, **kwargs):
        if torch.rand(1) < self.p:
            kwargs['mask'] = F.hflip(kwargs['mask'])
            return F.hflip(img), kwargs
        return img, kwargs


@TRANSFORM_REGISTRY.register()
class PairResize(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.resize = Resize(size=size)

    def forward(self, img, **kwargs):
        kwargs['mask'] = self.resize(kwargs['mask'])
        img = self.resize(img)
        return img, kwargs


class PairCompose(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, img, **kwargs):
        for t in self.transforms:
            if len(inspect.signature(t.forward).parameters
                   ) == 1:  # count how many args, not count self
                img = t(img)
            else:
                img, kwargs = t(img, **kwargs)
        return img, kwargs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


@TRANSFORM_REGISTRY.register()
class HumanResizeCropFinalV3(nn.Module):
    def __init__(self, size, crop_p=0.5):
        super().__init__()
        self.size = size
        self.crop_p = crop_p
        self.random_crop = RandomCrop(size=size)
        self.paired_random_crop = PairRandomCrop(size=size)

    def forward(self, img, **kwargs):
        # step 1: short edge resize to 512
        img = F.resize(img, size=self.size)
        if 'mask' in kwargs:
            kwargs['mask'] = F.resize(kwargs['mask'], size=self.size)

        # step 2: random crop
        width, height = img.size
        if random.random() < self.crop_p:
            if height > width:
                crop_pos = random.randint(0, height - width)
                img = F.crop(img, 0, 0, width + crop_pos, width)
                if 'mask' in kwargs:
                    kwargs['mask'] = F.crop(kwargs['mask'], 0, 0, width + crop_pos, width)
            else:
                if 'mask' in kwargs:
                    img, kwargs = self.paired_random_crop(img, **kwargs)
                else:
                    img = self.random_crop(img)
        else:
            img = img

        # step 3: long edge resize
        img = F.resize(img, size=self.size - 1, max_size=self.size)
        if 'mask' in kwargs:
            kwargs['mask'] = F.resize(kwargs['mask'], size=self.size - 1, max_size=self.size)

        new_width, new_height = img.size

        img = np.array(img)
        if 'mask' in kwargs:
            kwargs['mask'] = np.array(kwargs['mask']) / 255

        start_y = random.randint(0, 512 - new_height)
        start_x = random.randint(0, 512 - new_width)

        res_img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        res_mask = np.zeros((self.size, self.size))
        res_img_mask = np.zeros((self.size, self.size))

        res_img[start_y:start_y + new_height, start_x:start_x + new_width, :] = img
        if 'mask' in kwargs:
            res_mask[start_y:start_y + new_height, start_x:start_x + new_width] = kwargs['mask']
            kwargs['mask'] = res_mask

        res_img_mask[start_y:start_y + new_height, start_x:start_x + new_width] = 1
        kwargs['img_mask'] = res_img_mask

        img = Image.fromarray(res_img)

        if 'mask' in kwargs:
            kwargs['mask'] = cv2.resize(kwargs['mask'], (self.size // 8, self.size // 8), cv2.INTER_NEAREST)
            kwargs['mask'] = torch.from_numpy(kwargs['mask'])
        kwargs['img_mask'] = cv2.resize(kwargs['img_mask'], (self.size // 8, self.size // 8), cv2.INTER_NEAREST)
        kwargs['img_mask'] = torch.from_numpy(kwargs['img_mask'])
        return img, kwargs


@TRANSFORM_REGISTRY.register()
class ResizeFillMaskNew(nn.Module):
    def __init__(self, size, crop_p, scale_ratio):
        super().__init__()
        self.size = size
        self.crop_p = crop_p
        self.scale_ratio = scale_ratio
        self.random_crop = RandomCrop(size=size)
        self.paired_random_crop = PairRandomCrop(size=size)

    def forward(self, img, **kwargs):
        # width, height = img.size

        # step 1: short edge resize to 512
        img = F.resize(img, size=self.size)
        if 'mask' in kwargs:
            kwargs['mask'] = F.resize(kwargs['mask'], size=self.size)

        # step 2: random crop
        if random.random() < self.crop_p:
            if 'mask' in kwargs:
                img, kwargs = self.paired_random_crop(img, **kwargs)  # 51
            else:
                img = self.random_crop(img)  # 512
        else:
            # long edge resize
            img = F.resize(img, size=self.size - 1, max_size=self.size)
            if 'mask' in kwargs:
                kwargs['mask'] = F.resize(kwargs['mask'], size=self.size - 1, max_size=self.size)

        # step 3: random aspect ratio
        width, height = img.size
        ratio = random.uniform(*self.scale_ratio)

        img = F.resize(img, size=(int(height * ratio), int(width * ratio)))
        if 'mask' in kwargs:
            kwargs['mask'] = F.resize(kwargs['mask'], size=(int(height * ratio), int(width * ratio)), interpolation=0)

        # step 4: random place
        new_width, new_height = img.size

        img = np.array(img)
        if 'mask' in kwargs:
            kwargs['mask'] = np.array(kwargs['mask']) / 255

        start_y = random.randint(0, 512 - new_height)
        start_x = random.randint(0, 512 - new_width)

        res_img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        res_mask = np.zeros((self.size, self.size))
        res_img_mask = np.zeros((self.size, self.size))

        res_img[start_y:start_y + new_height, start_x:start_x + new_width, :] = img
        if 'mask' in kwargs:
            res_mask[start_y:start_y + new_height, start_x:start_x + new_width] = kwargs['mask']
            kwargs['mask'] = res_mask

        res_img_mask[start_y:start_y + new_height, start_x:start_x + new_width] = 1
        kwargs['img_mask'] = res_img_mask

        img = Image.fromarray(res_img)

        if 'mask' in kwargs:
            kwargs['mask'] = cv2.resize(kwargs['mask'], (self.size // 8, self.size // 8), cv2.INTER_NEAREST)
            kwargs['mask'] = torch.from_numpy(kwargs['mask'])
        kwargs['img_mask'] = cv2.resize(kwargs['img_mask'], (self.size // 8, self.size // 8), cv2.INTER_NEAREST)
        kwargs['img_mask'] = torch.from_numpy(kwargs['img_mask'])

        return img, kwargs


@TRANSFORM_REGISTRY.register()
class ShuffleCaption(nn.Module):
    def __init__(self, keep_token_num):
        super().__init__()
        self.keep_token_num = keep_token_num

    def forward(self, img, **kwargs):
        prompts = kwargs['prompts'].strip()

        fixed_tokens = []
        flex_tokens = [t.strip() for t in prompts.strip().split(',')]
        if self.keep_token_num > 0:
            fixed_tokens = flex_tokens[:self.keep_token_num]
            flex_tokens = flex_tokens[self.keep_token_num:]

        random.shuffle(flex_tokens)
        prompts = ', '.join(fixed_tokens + flex_tokens)
        kwargs['prompts'] = prompts
        return img, kwargs


@TRANSFORM_REGISTRY.register()
class EnhanceText(nn.Module):
    def __init__(self, enhance_type='object'):
        super().__init__()
        STYLE_TEMPLATE = [
            'a painting in the style of {}',
            'a rendering in the style of {}',
            'a cropped painting in the style of {}',
            'the painting in the style of {}',
            'a clean painting in the style of {}',
            'a dirty painting in the style of {}',
            'a dark painting in the style of {}',
            'a picture in the style of {}',
            'a cool painting in the style of {}',
            'a close-up painting in the style of {}',
            'a bright painting in the style of {}',
            'a cropped painting in the style of {}',
            'a good painting in the style of {}',
            'a close-up painting in the style of {}',
            'a rendition in the style of {}',
            'a nice painting in the style of {}',
            'a small painting in the style of {}',
            'a weird painting in the style of {}',
            'a large painting in the style of {}',
        ]

        OBJECT_TEMPLATE = [
            'a photo of a {}',
            'a rendering of a {}',
            'a cropped photo of the {}',
            'the photo of a {}',
            'a photo of a clean {}',
            'a photo of a dirty {}',
            'a dark photo of the {}',
            'a photo of my {}',
            'a photo of the cool {}',
            'a close-up photo of a {}',
            'a bright photo of the {}',
            'a cropped photo of a {}',
            'a photo of the {}',
            'a good photo of the {}',
            'a photo of one {}',
            'a close-up photo of the {}',
            'a rendition of the {}',
            'a photo of the clean {}',
            'a rendition of a {}',
            'a photo of a nice {}',
            'a good photo of a {}',
            'a photo of the nice {}',
            'a photo of the small {}',
            'a photo of the weird {}',
            'a photo of the large {}',
            'a photo of a cool {}',
            'a photo of a small {}',
        ]

        HUMAN_TEMPLATE = [
            'a photo of a {}', 'a photo of one {}', 'a photo of the {}',
            'the photo of a {}', 'a rendering of a {}',
            'a rendition of the {}', 'a rendition of a {}',
            'a cropped photo of the {}', 'a cropped photo of a {}',
            'a bad photo of the {}', 'a bad photo of a {}',
            'a photo of a weird {}', 'a weird photo of a {}',
            'a bright photo of the {}', 'a good photo of the {}',
            'a photo of a nice {}', 'a good photo of a {}',
            'a photo of a cool {}', 'a bright photo of the {}'
        ]

        if enhance_type == 'object':
            self.templates = OBJECT_TEMPLATE
        elif enhance_type == 'style':
            self.templates = STYLE_TEMPLATE
        elif enhance_type == 'human':
            self.templates = HUMAN_TEMPLATE
        else:
            raise NotImplementedError

    def forward(self, img, **kwargs):
        concept_token = kwargs['prompts'].strip()
        kwargs['prompts'] = random.choice(self.templates).format(concept_token)
        return img, kwargs
