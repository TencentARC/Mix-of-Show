import argparse
import copy
import hashlib
import os.path
import torch
from diffusers import Adapter, StableDiffusionAdapterPipeline, StableDiffusionPipeline
from PIL import Image

from mixofshow.archs.edlora_override import revise_unet_attention_forward
from mixofshow.utils.diffusers_sample_util import StableDiffusion_PPlus_Sample, StableDiffusion_Sample
from mixofshow.utils.regionally_controllable_sample_util import Regionally_T2IAdaptor_Sample


def inference_image(pipe,
                    input_prompt,
                    input_neg_prompt=None,
                    generator=None,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    sketch_adaptor_weight=1.0,
                    region_sketch_adaptor_weight='',
                    keypose_adaptor_weight=1.0,
                    region_keypose_adaptor_weight='',
                    pipeline_type='sd',
                    **extra_kargs):
    if pipeline_type == 'adaptor_pplus' or pipeline_type == 'adaptor_sd':
        keypose_condition = extra_kargs.pop('keypose_condition')
        if keypose_condition is not None:
            keypose_adapter_input = [keypose_condition] * len(input_prompt)
        else:
            keypose_adapter_input = None

        sketch_condition = extra_kargs.pop('sketch_condition')
        if sketch_condition is not None:
            sketch_adapter_input = [sketch_condition] * len(input_prompt)
        else:
            sketch_adapter_input = None

        new_concept_cfg = extra_kargs.pop('new_concept_cfg')
        images = Regionally_T2IAdaptor_Sample(
            pipe,
            prompt=input_prompt,
            negative_prompt=input_neg_prompt,
            new_concept_cfg=new_concept_cfg,
            keypose_adapter_input=keypose_adapter_input,
            keypose_adaptor_weight=keypose_adaptor_weight,
            region_keypose_adaptor_weight=region_keypose_adaptor_weight,
            sketch_adapter_input=sketch_adapter_input,
            sketch_adaptor_weight=sketch_adaptor_weight,
            region_sketch_adaptor_weight=region_sketch_adaptor_weight,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            **extra_kargs).images
    elif pipeline_type == 'sd_pplus':
        new_concept_cfg = extra_kargs.pop('new_concept_cfg')
        images = StableDiffusion_PPlus_Sample(
            pipe,
            prompt=input_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=input_neg_prompt,
            generator=generator,
            new_concept_cfg=new_concept_cfg,
        ).images
    elif pipeline_type == 'sd':
        images = StableDiffusion_Sample(
            pipe,
            prompt=input_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=input_neg_prompt,
            generator=generator,
        ).images
    else:
        raise NotImplementedError
    return images


def merge_pplus2sd_(pipe, lora_weight_path):

    def add_new_concept(embedding):
        new_token_names = [f'<new{start_idx + layer_id}>' for layer_id in range(NUM_CROSS_ATTENTION_LAYERS)]
        num_added_tokens = tokenizer.add_tokens(new_token_names)
        assert num_added_tokens == NUM_CROSS_ATTENTION_LAYERS
        new_token_ids = [tokenizer.convert_tokens_to_ids(token_name) for token_name in new_token_names]

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data

        token_embeds[new_token_ids] = token_embeds[new_token_ids].copy_(embedding)
        return start_idx + NUM_CROSS_ATTENTION_LAYERS, new_token_ids, new_token_names

    # step 1: list pipe module
    tokenizer, unet, text_encoder = pipe.tokenizer, pipe.unet, pipe.text_encoder
    lora_weight = torch.load(lora_weight_path, map_location='cpu')['params']
    new_concept_cfg = {}

    # step 2: load embedding into tokenizer/text_encoder:
    if 'new_concept_embedding' in lora_weight and len(lora_weight['new_concept_embedding']) != 0:
        start_idx = 0
        NUM_CROSS_ATTENTION_LAYERS = 16
        for idx, (concept_name, embedding) in enumerate(lora_weight['new_concept_embedding'].items()):
            start_idx, new_token_ids, new_token_names = add_new_concept(embedding)
            new_concept_cfg.update(
                {concept_name: {
                    'concept_token_ids': new_token_ids,
                    'concept_token_names': new_token_names
                }})

    # step 3: load text_encoder_weight:
    if 'text_encoder' in lora_weight and len(lora_weight['text_encoder']) != 0:
        sd_textenc_state_dict = copy.deepcopy(text_encoder.state_dict())

        for k in lora_weight['text_encoder'].keys():
            sd_textenc_state_dict[k] = lora_weight['text_encoder'][k]
        text_encoder.load_state_dict(sd_textenc_state_dict)

    if 'unet' in lora_weight and len(lora_weight['unet']) != 0:
        sd_unet_state_dict = copy.deepcopy(unet.state_dict())

        for k in lora_weight['unet'].keys():
            sd_unet_state_dict[k] = lora_weight['unet'][k]

        unet.load_state_dict(sd_unet_state_dict)
    return new_concept_cfg


def merge2sd_(pipe, lora_weight_path):

    # step 1: list pipe module
    tokenizer, unet, text_encoder = pipe.tokenizer, pipe.unet, pipe.text_encoder
    lora_weight = torch.load(lora_weight_path, map_location='cpu')['params']

    # step 2: load embedding into tokenizer/text_encoder:
    if 'new_concept_embedding' in lora_weight and len(lora_weight['new_concept_embedding']) != 0:
        new_concept_embedding = list(lora_weight['new_concept_embedding'].keys())

        for new_token in new_concept_embedding:
            # Add the placeholder token in tokenizer
            _ = tokenizer.add_tokens(new_token)
            new_token_id = tokenizer.convert_tokens_to_ids(new_token)
            text_encoder.resize_token_embeddings(len(tokenizer))
            token_embeds = text_encoder.get_input_embeddings().weight.data
            token_embeds[new_token_id] = lora_weight['new_concept_embedding'][new_token]

    # step 3: load text_encoder_weight:
    if 'text_encoder' in lora_weight and len(lora_weight['text_encoder']) != 0:
        sd_textenc_state_dict = copy.deepcopy(text_encoder.state_dict())

        for k in lora_weight['text_encoder'].keys():
            sd_textenc_state_dict[k] = lora_weight['text_encoder'][k]
        text_encoder.load_state_dict(sd_textenc_state_dict)

    if 'unet' in lora_weight and len(lora_weight['unet']) != 0:
        sd_unet_state_dict = copy.deepcopy(unet.state_dict())

        for k in lora_weight['unet'].keys():
            sd_unet_state_dict[k] = lora_weight['unet'][k]

        unet.load_state_dict(sd_unet_state_dict)


def build_model(pretrained_model, combined_model, sketch_adaptor_model, keypose_adaptor_model, pipeline_type, device):

    if pipeline_type == 'adaptor_pplus' or pipeline_type == 'adaptor_sd':
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            pretrained_model, torch_dtype=torch.float16, safety_checker=None).to(device)

        if pipeline_type == 'adaptor_pplus':
            new_concept_cfg = merge_pplus2sd_(pipe, lora_weight_path=combined_model)
        else:
            merge2sd_(pipe, lora_weight_path=combined_model)
            new_concept_cfg = None

        pipe.keypose_adapter = Adapter(
            cin=int(3 * 64), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).half()
        pipe.keypose_adapter.load_state_dict(torch.load(keypose_adaptor_model))
        pipe.keypose_adapter = pipe.keypose_adapter.to(device)

        pipe.sketch_adapter = Adapter(
            cin=int(64), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).half()
        pipe.sketch_adapter.load_state_dict(torch.load(sketch_adaptor_model))
        pipe.sketch_adapter = pipe.sketch_adapter.to(device)
    elif pipeline_type == 'sd':
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model, torch_dtype=torch.float16, safety_checker=None).to('cuda')
        merge2sd_(pipe, lora_weight_path=combined_model)
        new_concept_cfg = None
    elif pipeline_type == 'sd_pplus':
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model, torch_dtype=torch.float16, safety_checker=None).to('cuda')
        revise_unet_attention_forward(pipe.unet)
        new_concept_cfg = merge_pplus2sd_(pipe, lora_weight_path=combined_model)
    else:
        raise NotImplementedError
    return pipe, new_concept_cfg


def prepare_text(prompt, region_prompts, height, width):
    '''
    Args:
        prompt_entity: [subject1]-*-[attribute1]-*-[Location1]|[subject2]-*-[attribute2]-*-[Location2]|[global text]
    Returns:
        full_prompt: subject1, attribute1 and subject2, attribute2, global text
        context_prompt: subject1 and subject2, global text
        entity_collection: [(subject1, attribute1), Location1]
    '''
    region_collection = []

    regions = region_prompts.split('|')

    for region in regions:
        if region == '':
            break
        prompt_region, neg_prompt_region, pos = region.split('-*-')
        prompt_region = prompt_region.replace('[', '').replace(']', '')
        neg_prompt_region = neg_prompt_region.replace('[', '').replace(']', '')
        pos = eval(pos)
        if len(pos) == 0:
            pos = [0, 0, 1, 1]
        else:
            pos[0], pos[2] = pos[0] / height, pos[2] / height
            pos[1], pos[3] = pos[1] / width, pos[3] / width

        region_collection.append((prompt_region, neg_prompt_region, pos))
    return (prompt, region_collection)


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--pretrained_model', default='experiments/pretrained_models/anything-v4.0', type=str)
    parser.add_argument(
        '--combined_model',
        default='experiments/pretrained_models/composed_optimized_logs/anythingv4/yangcai+kaori/combined_model.ckpt',
        type=str)
    parser.add_argument('--sketch_adaptor_model', default=None, type=str)
    parser.add_argument('--sketch_condition', default=None, type=str)
    parser.add_argument('--sketch_adaptor_weight', default=1.0, type=float)
    parser.add_argument('--region_sketch_adaptor_weight', default='', type=str)
    parser.add_argument('--keypose_adaptor_model', default=None, type=str)
    parser.add_argument('--keypose_condition', default=None, type=str)
    parser.add_argument('--keypose_adaptor_weight', default=1.0, type=float)
    parser.add_argument('--region_keypose_adaptor_weight', default='', type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--pipeline_type', default='sd', type=str)
    parser.add_argument('--prompt', default='photo of a toy', type=str)
    parser.add_argument('--negative_prompt', default='', type=str)
    parser.add_argument('--prompt_rewrite', default='', type=str)
    parser.add_argument('--seed', default=16141, type=int)
    parser.add_argument('--suffix', default='', type=str)
    parser.add_argument('--no_region', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe, new_concept_cfg = build_model(args.pretrained_model, args.combined_model, args.sketch_adaptor_model,
                                        args.keypose_adaptor_model, args.pipeline_type, device)

    prompts = [args.prompt]
    prompts_rewrite = [''] if args.no_region else [args.prompt_rewrite]

    if args.pipeline_type == 'adaptor_pplus':
        if args.sketch_condition is not None and os.path.exists(args.sketch_condition):
            sketch_condition = Image.open(args.sketch_condition).convert('L')
            width_sketch, height_sketch = sketch_condition.size
            print('use sketch condition')
        else:
            sketch_condition, width_sketch, height_sketch = None, 0, 0
            print('skip sketch condition')

        if args.keypose_condition is not None and os.path.exists(args.keypose_condition):
            keypose_condition = Image.open(args.keypose_condition).convert('RGB')
            width_pose, height_pose = keypose_condition.size
            print('use pose condition')
        else:
            keypose_condition, width_pose, height_pose = None, 0, 0
            print('skip pose condition')

        if width_sketch != 0 and width_pose != 0:
            assert width_sketch == width_pose and height_sketch == height_pose, 'conditions should be same size'
        width, height = max(width_pose, width_sketch), max(height_pose, height_sketch)

        kwargs = {
            'sketch_condition': sketch_condition,
            'keypose_condition': keypose_condition,
            'new_concept_cfg': new_concept_cfg,
            'height': height,
            'width': width,
        }
        input_prompt = [prepare_text(p, p_w, height, width) for p, p_w in zip(prompts, prompts_rewrite)]
        save_prompt = input_prompt[0][0]
    elif args.pipeline_type == 'sd':
        kwargs = {
            'new_concept_cfg': new_concept_cfg,
            'height': 512,
            'width': 512,
        }
        input_prompt = [args.prompt]
        save_prompt = input_prompt[0]
    else:
        raise NotImplementedError

    image = inference_image(
        pipe,
        input_prompt=input_prompt,
        input_neg_prompt=[args.negative_prompt] * len(input_prompt),
        generator=torch.Generator(device).manual_seed(args.seed),
        sketch_adaptor_weight=args.sketch_adaptor_weight,
        region_sketch_adaptor_weight=args.region_sketch_adaptor_weight,
        keypose_adaptor_weight=args.keypose_adaptor_weight,
        region_keypose_adaptor_weight=args.region_keypose_adaptor_weight,
        pipeline_type=args.pipeline_type,
        **kwargs)

    print(f'save to: {args.save_dir}')

    configs = [
        f'pretrained_model: {args.pretrained_model}\n', f'combined_model: {args.combined_model}\n',
        f'context_prompt: {args.prompt}\n', f'neg_context_prompt: {args.negative_prompt}\n',
        f'sketch_condition: {args.sketch_condition}\n', f'sketch_adaptor_weight: {args.sketch_adaptor_weight}\n',
        f'region_sketch_adaptor_weight: {args.region_sketch_adaptor_weight}\n',
        f'keypose_condition: {args.keypose_condition}\n', f'keypose_adaptor_weight: {args.keypose_adaptor_weight}\n',
        f'region_keypose_adaptor_weight: {args.region_keypose_adaptor_weight}\n', f'random seed: {args.seed}\n',
        f'prompt_rewrite: {args.prompt_rewrite}\n'
    ]
    hash_code = hashlib.sha256(''.join(configs).encode('utf-8')).hexdigest()[:8]

    save_prompt = save_prompt.replace(' ', '_')
    save_name = f'{save_prompt}---{args.suffix}---{hash_code}.png'
    save_dir = os.path.join(args.save_dir, f'seed_{args.seed}')
    save_path = os.path.join(save_dir, save_name)
    save_config_path = os.path.join(save_dir, save_name.replace('.png', '.txt'))

    os.makedirs(save_dir, exist_ok=True)
    image[0].save(os.path.join(save_dir, save_name))

    with open(save_config_path, 'w') as fw:
        fw.writelines(configs)
