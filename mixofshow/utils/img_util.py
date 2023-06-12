import os
import os.path
import os.path as osp
import PIL
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.transforms import ToTensor
from torchvision.utils import make_grid

from mixofshow.utils.dist_util import master_only


def pil_imwrite(img, file_path, auto_mkdir=True):
    """Write image to file.
    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.
    Returns:
        bool: Successful or not.
    """
    assert isinstance(img, PIL.Image.Image), 'model should return a list of PIL images'
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    img.save(file_path)


def draw_prompt(text, height, width, font_size=45):
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(osp.join(osp.dirname(osp.abspath(__file__)), 'arial.ttf'), font_size)

    guess_count = 0

    while font.font.getsize(
            text[:guess_count])[0][0] + 0.1 * width < width - 0.1 * width and guess_count < len(text):  # centerize
        guess_count += 1

    text_new = ''
    for idx, s in enumerate(text):
        if idx % guess_count == 0:
            text_new += '\n'
            if s == ' ':
                s = ''  # new line trip the first space
        text_new += s

    draw.text([int(0.1 * width), int(0.3 * height)], text_new, font=font, fill='black')
    return img


@master_only
def compose_visualize(dir_path):
    file_list = sorted(os.listdir(dir_path))
    img_list = []
    info_dict = {'prompts': set(), 'sample_args': set(), 'suffix': set()}
    for filename in file_list:
        prompt, sample_args, index, suffix = osp.splitext(osp.basename(filename))[0].split('---')

        filepath = osp.join(dir_path, filename)
        img = ToTensor()(Image.open(filepath))
        height, width = img.shape[1:]

        if prompt not in info_dict['prompts']:
            img_list.append(ToTensor()(draw_prompt(prompt, height=height, width=width, font_size=45)))
        info_dict['prompts'].add(prompt)
        info_dict['sample_args'].add(sample_args)
        info_dict['suffix'].add(suffix)

        img_list.append(img)
    assert len(info_dict['sample_args']) == 1, 'compose dir should contain images form same sample args.'
    assert len(info_dict['suffix']) == 1, 'compose dir should contain images form same suffix.'

    grid = make_grid(img_list, nrow=len(img_list) // len(info_dict['prompts']))
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    save_name = f"{info_dict['sample_args'].pop()}---{info_dict['suffix'].pop()}.jpg"
    im.save(osp.join(osp.dirname(dir_path), save_name))
