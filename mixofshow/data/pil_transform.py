import inspect
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
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
class ResizeCrop_InstanceMask(nn.Module):
    def __init__(self, size, scale_ratio=[0.75, 1]):
        super().__init__()
        self.size = size
        self.scale_ratio = scale_ratio

    def make_divisible_to_64(self, image):
        width, height = image.size

        new_width = (width // 64) * 64
        new_height = (height // 64) * 64

        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = (width + new_width) // 2
        bottom = (height + new_height) // 2

        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image

    def forward(self, img, **kwargs):
        # step 1: short edge resize to 512
        img = F.resize(img, size=self.size)

        if 'mask' in kwargs:
            kwargs['mask'] = F.resize(kwargs['mask'], size=self.size)

        # max ratio set to 1:2
        width, height = img.size
        if height > width and height > 2 * self.size:
            img = F.crop(img, 0, 0, 2 * self.size, width)
            if 'mask' in kwargs:
                kwargs['mask'] = F.crop(kwargs['mask'], 0, 0, 2 * self.size, width)
        elif width > height and width > 2 * self.size:
            img = F.center_crop(img=img, output_size=(height, 2 * self.size))
            if 'mask' in kwargs:
                kwargs['mask'] = F.center_crop(kwargs['mask'], output_size=(height, 2 * self.size))

        # random scale
        ratio = random.uniform(*self.scale_ratio)
        width, height = img.size
        img = F.resize(img, size=(int(height * ratio), int(width * ratio)))
        if 'mask' in kwargs:
            kwargs['mask'] = F.resize(kwargs['mask'], size=(int(height * ratio), int(width * ratio)))

        # make divisible to 8
        img = self.make_divisible_to_64(img)
        if 'mask' in kwargs:
            kwargs['mask'] = self.make_divisible_to_64(kwargs['mask'])

        width, height = img.size

        if 'mask' in kwargs:
            mask = np.array(kwargs['mask']) / 255
        else:
            mask = np.ones((height, width))

        kwargs['mask'] = cv2.resize(mask, (width // 8, height // 8), cv2.INTER_NEAREST)
        kwargs['mask'] = torch.from_numpy(kwargs['mask'])
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
