import argparse
import hashlib
import torch
from diffusers import DiffusionPipeline
from pathlib import Path
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class PromptDataset(Dataset):
    'A simple dataset to prepare the prompts to generate class images on multiple GPUs.'

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example['prompt'] = self.prompt
        example['index'] = index
        return example


def synthesize(class_prompt, class_data_dir, model_name, num_class_images):
    class_images_dir = Path(class_data_dir)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))

    if cur_class_images < num_class_images:
        torch_dtype = torch.float32
        pipeline = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            safety_checker=None,
            revision=None,
        )
        pipeline.set_progress_bar_config(disable=True)

        num_new_images = num_class_images - cur_class_images
        print(f'Number of class images to sample: {num_new_images}.')

        sample_dataset = PromptDataset(class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=4)

        pipeline = pipeline.to('cuda')

        for example in tqdm(sample_dataloader, desc='Generating class images'):
            images = pipeline(example['prompt']).images

            for i, image in enumerate(images):
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--class_prompt', help='prompt to generate class images', default='photo of a man', type=str)
    parser.add_argument(
        '--class_data_dir',
        help='dir to class images',
        default='./datasets/Dreambooth_Regularization/character/real/man',
        type=str)
    parser.add_argument(
        '--model_name',
        help='stable_diffusion model path',
        default='experiments/pretrained_models/chilloutmix',
        type=str)
    parser.add_argument('--num_class_images', help='number of retrieved images', default=200, type=int)
    return parser.parse_args()
    # python scripts/generate_prompt.py --target_name "wooden pot"


if __name__ == '__main__':
    args = parse_args()
    synthesize(args.class_prompt, args.class_data_dir, args.model_name, args.num_class_images)
