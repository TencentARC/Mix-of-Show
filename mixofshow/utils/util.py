import datetime
import logging
import os
import os.path
import os.path as osp
import time
from collections import OrderedDict

import PIL
import torch
from accelerate.logging import get_logger
from accelerate.state import PartialState
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.transforms import ToTensor
from torchvision.utils import make_grid

NEGATIVE_PROMPT = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'


# ----------- file/logger util ----------
def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        mkdir_and_rename(path_opt.pop('experiments_root'))
    else:
        mkdir_and_rename(path_opt.pop('results_root'))
    for key, path in path_opt.items():
        if ('strict_load' in key) or ('pretrain_network' in key) or (
                'resume' in key) or ('param_key' in key) or ('lora_path' in key):
            continue
        else:
            os.makedirs(path, exist_ok=True)


def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile
    cmd = ' '.join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    copyfile(opt_file, filename)

    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines.insert(
            0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
        f.seek(0)
        f.writelines(lines)


def set_path_logger(accelerator, root_path, config_path, opt, is_train=True):
    opt['is_train'] = is_train

    if is_train:
        experiments_root = osp.join(root_path, 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')
    else:
        results_root = osp.join(root_path, 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    # Handle the output folder creation
    if accelerator.is_main_process:
        make_exp_dirs(opt)

    accelerator.wait_for_everyone()

    if is_train:
        copy_opt_file(config_path, opt['path']['experiments_root'])
        log_file = osp.join(opt['path']['log'],
                            f"train_{opt['name']}_{get_time_str()}.log")
        set_logger(log_file)
    else:
        copy_opt_file(config_path, opt['path']['results_root'])
        log_file = osp.join(opt['path']['log'],
                            f"test_{opt['name']}_{get_time_str()}.log")
        set_logger(log_file)


def set_logger(log_file=None):
    # Make one log on every process with the configuration for debugging.
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    log_level = logging.INFO
    handlers = []

    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(logging.Formatter(format_str))
    file_handler.setLevel(log_level)
    handlers.append(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    handlers.append(stream_handler)

    logging.basicConfig(handlers=handlers, level=log_level)


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


class MessageLogger():
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
    """
    def __init__(self, opt, start_iter=1):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.start_time = time.time()
        self.logger = get_logger('mixofshow', log_level='INFO')

    def reset_start_time(self):
        self.start_time = time.time()

    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # epoch, iter, learning rates
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')

        message = (
            f'[{self.exp_name[:5]}..][Iter:{current_iter:8,d}, lr:('
        )
        for v in lrs:
            message += f'{v:.3e},'
        message += ')] '

        # time and estimated time
        total_time = time.time() - self.start_time
        time_sec_avg = total_time / (current_iter - self.start_iter + 1)
        eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        message += f'[eta: {eta_str}] '

        # other items, especially losses
        for k, v in log_vars.items():
            message += f'{k}: {v:.4e} '

        self.logger.info(message)


def reduce_loss_dict(accelerator, loss_dict):
    """reduce loss dict.

    In distributed training, it averages the losses among different GPUs .

    Args:
        loss_dict (OrderedDict): Loss dict.
    """
    with torch.no_grad():
        keys = []
        losses = []
        for name, value in loss_dict.items():
            keys.append(name)
            losses.append(value)
        losses = torch.stack(losses, 0)
        losses = accelerator.reduce(losses)

        world_size = PartialState().num_processes
        losses /= world_size

        loss_dict = {key: loss for key, loss in zip(keys, losses)}

        log_dict = OrderedDict()
        for name, value in loss_dict.items():
            log_dict[name] = value.mean().item()

        return log_dict


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
    assert isinstance(
        img, PIL.Image.Image), 'model should return a list of PIL images'
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    img.save(file_path)


def draw_prompt(text, height, width, font_size=45):
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(
        osp.join(osp.dirname(osp.abspath(__file__)), 'arial.ttf'), font_size)

    guess_count = 0

    while font.font.getsize(text[:guess_count])[0][
            0] + 0.1 * width < width - 0.1 * width and guess_count < len(
                text):  # centerize
        guess_count += 1

    text_new = ''
    for idx, s in enumerate(text):
        if idx % guess_count == 0:
            text_new += '\n'
            if s == ' ':
                s = ''  # new line trip the first space
        text_new += s

    draw.text([int(0.1 * width), int(0.3 * height)],
              text_new,
              font=font,
              fill='black')
    return img


def compose_visualize(dir_path):
    file_list = sorted(os.listdir(dir_path))
    img_list = []
    info_dict = {'prompts': set(), 'sample_args': set(), 'suffix': set()}
    for filename in file_list:
        prompt, sample_args, index, suffix = osp.splitext(
            osp.basename(filename))[0].split('---')

        filepath = osp.join(dir_path, filename)
        img = ToTensor()(Image.open(filepath))
        height, width = img.shape[1:]

        if prompt not in info_dict['prompts']:
            img_list.append(ToTensor()(draw_prompt(prompt,
                                                   height=height,
                                                   width=width,
                                                   font_size=45)))
        info_dict['prompts'].add(prompt)
        info_dict['sample_args'].add(sample_args)
        info_dict['suffix'].add(suffix)

        img_list.append(img)
    assert len(
        info_dict['sample_args']
    ) == 1, 'compose dir should contain images form same sample args.'
    assert len(info_dict['suffix']
               ) == 1, 'compose dir should contain images form same suffix.'

    grid = make_grid(img_list, nrow=len(img_list) // len(info_dict['prompts']))
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(
        'cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    save_name = f"{info_dict['sample_args'].pop()}---{info_dict['suffix'].pop()}.jpg"
    im.save(osp.join(osp.dirname(dir_path), save_name))
