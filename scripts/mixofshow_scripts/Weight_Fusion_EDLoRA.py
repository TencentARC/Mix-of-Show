import argparse
import copy
import itertools
import json
import logging
import os
import torch
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline

from mixofshow.utils import get_root_logger


def merge_lora_into_weight(original_state_dict, lora_state_dict, modification_layer_names, model_type, alpha, device):

    def get_lora_down_name(original_layer_name):
        if model_type == 'text_encoder':
            lora_down_name = original_layer_name.replace('q_proj.weight', 'q_proj_lora.down.weight') \
                .replace('k_proj.weight', 'k_proj_lora.down.weight') \
                .replace('v_proj.weight', 'v_proj_lora.down.weight') \
                .replace('out_proj.weight', 'out_proj_lora.down.weight') \
                .replace('fc1.weight', 'fc1_lora.down.weight') \
                .replace('fc2.weight', 'fc2_lora.down.weight')
        else:
            lora_down_name = k.replace('to_q.weight', 'to_q_lora.down.weight') \
                .replace('to_k.weight', 'to_k_lora.down.weight') \
                .replace('to_v.weight', 'to_v_lora.down.weight') \
                .replace('to_out.0.weight', 'to_out.0_lora.down.weight') \
                .replace('ff.net.0.proj.weight', 'ff.net.0.proj_lora.down.weight') \
                .replace('ff.net.2.weight', 'ff.net.2_lora.down.weight') \
                .replace('proj_out.weight', 'proj_out_lora.down.weight') \
                .replace('proj_in.weight', 'proj_in_lora.down.weight')

        return lora_down_name

    logger = get_root_logger()
    assert model_type in ['unet', 'text_encoder']
    new_state_dict = original_state_dict

    load_cnt = 0
    for k in modification_layer_names:
        lora_down_name = get_lora_down_name(k)
        lora_up_name = lora_down_name.replace('lora.down', 'lora.up')

        if lora_up_name in lora_state_dict:
            load_cnt += 1
            original_params = new_state_dict[k]
            lora_down_params = lora_state_dict[lora_down_name].to(device)
            lora_up_params = lora_state_dict[lora_up_name].to(device)
            if len(original_params.shape) == 4:
                lora_param = lora_up_params.squeeze() @ lora_down_params.squeeze()
                lora_param = lora_param.unsqueeze(-1).unsqueeze(-1)
            else:
                lora_param = lora_up_params @ lora_down_params
            merge_params = original_params + alpha * lora_param
            new_state_dict[k] = merge_params

    logger.info(f'merge {load_cnt} LoRAs of {model_type}')
    return new_state_dict


def init_stable_diffusion(pretrained_model_path, device):
    # step1: get w0 parameters
    model_id = pretrained_model_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    train_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder='scheduler')
    test_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe.safety_checker = None
    pipe.scheduler = test_scheduler
    return pipe, train_scheduler, test_scheduler


def merge_new_concepts_edlora_(embedding_list, concept_list, tokenizer, text_encoder):

    def add_new_concept(concept_name, embedding):
        new_token_names = [f'<new{start_idx + layer_id}>' for layer_id in range(NUM_CROSS_ATTENTION_LAYERS)]
        num_added_tokens = tokenizer.add_tokens(new_token_names)
        assert num_added_tokens == NUM_CROSS_ATTENTION_LAYERS
        new_token_ids = [tokenizer.convert_tokens_to_ids(token_name) for token_name in new_token_names]

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data

        token_embeds[new_token_ids] = token_embeds[new_token_ids].copy_(embedding[concept_name])

        embedding_features.update({concept_name: embedding[concept_name]})
        logger.info(f'concept {concept_name} is bind with token_id: [{min(new_token_ids)}, {max(new_token_ids)}]')

        return start_idx + NUM_CROSS_ATTENTION_LAYERS, new_token_ids, new_token_names

    logger = get_root_logger()
    embedding_features = {}
    new_concept_cfg = {}

    start_idx = 0

    NUM_CROSS_ATTENTION_LAYERS = 16

    for idx, (embedding, concept) in enumerate(zip(embedding_list, concept_list)):
        concept_names = concept['concept_name'].split(' ')

        for concept_name in concept_names:
            if not concept_name.startswith('<'):
                continue
            else:
                assert concept_name in embedding, 'check the config, the provide concept name is not in the lora model'
            start_idx, new_token_ids, new_token_names = add_new_concept(concept_name, embedding)
            new_concept_cfg.update(
                {concept_name: {
                    'concept_token_ids': new_token_ids,
                    'concept_token_names': new_token_names
                }})
    return embedding_features, new_concept_cfg


def merge_new_concepts_lora_(embedding_list, concept_list, tokenizer, text_encoder):
    logger = get_root_logger()
    embedding_features = {}
    for embedding, concept in zip(embedding_list, concept_list):
        # composition of models with individual concept only
        for embed_name in embedding.keys():
            token_name = embed_name
            _ = tokenizer.add_tokens(token_name)
            new_token_id = tokenizer.convert_tokens_to_ids(token_name)
            text_encoder.resize_token_embeddings(len(tokenizer))
            token_embeds = text_encoder.get_input_embeddings().weight.data
            token_embeds[new_token_id] = embedding[embed_name]
            embedding_features[token_name] = embedding[embed_name]
            logger.info(f'concept {token_name} is bind with token_id: {new_token_id}')
    return embedding_features, None


def parse_new_concepts(concept_cfg):
    with open(concept_cfg, 'r') as f:
        concept_list = json.load(f)

    model_paths = [concept['lora_path'] for concept in concept_list]

    embedding_list = []
    text_encoder_list = []
    unet_list = []

    for model_path in model_paths:
        model = torch.load(model_path)['params']
        if 'new_concept_embedding' in model and len(model['new_concept_embedding']) != 0:
            embedding_list.append(model['new_concept_embedding'])
        elif 'new_concept_token' in model and len(model['new_concept_token']) != 0:
            embedding_list.append(model['new_concept_token'])
        else:
            embedding_list.append(None)

        if 'text_encoder' in model and len(model['text_encoder']) != 0:
            text_encoder_list.append(model['text_encoder'])
        else:
            text_encoder_list.append(None)

        if 'unet' in model and len(model['unet']) != 0:
            unet_list.append(model['unet'])
        else:
            unet_list.append(None)

    return embedding_list, text_encoder_list, unet_list, concept_list


def merge_unet(concept_list, unet, unet_list, device):

    logger = get_root_logger()

    LoRA_keys = []
    for textenc_lora in unet_list:
        LoRA_keys += list(textenc_lora.keys())
    LoRA_keys = set([key.replace('_lora.down', '').replace('_lora.up', '') for key in LoRA_keys])
    unet_layer_names = LoRA_keys

    logger.info(f'unet have {len(unet_layer_names)} linear layer need to optimize')

    merged_state_dict = copy.deepcopy(unet.state_dict())

    for concept, lora_state_dict in zip(concept_list, unet_list):
        merged_state_dict = merge_lora_into_weight(
            merged_state_dict,
            lora_state_dict,
            unet_layer_names,
            model_type='unet',
            alpha=concept['unet_alpha'] / len(concept_list),
            device=device)
    return merged_state_dict


def merge_text_encoder(concept_list, text_encoder, text_encoder_list, device):
    logger = get_root_logger()

    LoRA_keys = []
    for textenc_lora in text_encoder_list:
        LoRA_keys += list(textenc_lora.keys())
    LoRA_keys = set([key.replace('_lora.down', '').replace('_lora.up', '') for key in LoRA_keys])
    text_encoder_layer_names = LoRA_keys

    logger.info(f'text_encoder have {len(text_encoder_layer_names)} linear layer need to optimize')

    merged_state_dict = copy.deepcopy(text_encoder.state_dict())  # original state dict

    for concept, lora_state_dict in zip(concept_list, text_encoder_list):
        merged_state_dict = merge_lora_into_weight(
            merged_state_dict,
            lora_state_dict,
            text_encoder_layer_names,
            model_type='text_encoder',
            alpha=concept['text_encoder_alpha'] / len(concept_list),
            device=device)
    return merged_state_dict


def compose_concepts(concept_cfg, pretrained_model_path, save_path, suffix, type, device):
    logger = get_root_logger()

    logger.info('------Step 1: load stable diffusion checkpoint------')
    pipe, _, _ = init_stable_diffusion(pretrained_model_path, device)
    tokenizer, text_encoder, unet, vae = pipe.tokenizer, pipe.text_encoder, pipe.unet, pipe.vae
    for param in itertools.chain(text_encoder.parameters(), unet.parameters(), vae.parameters()):
        param.requires_grad = False

    logger.info('------Step 2: load new concepts checkpoints------')
    embedding_list, text_encoder_list, unet_list, concept_list = parse_new_concepts(concept_cfg)

    # step 1: inplace add new concept to tokenizer and embedding layers of text encoder
    if any([item is not None for item in embedding_list]):
        logger.info('------Step 3: merge token embedding------')
        if type == 'lora':
            merge_new_concepts_ = merge_new_concepts_lora_
        else:
            merge_new_concepts_ = merge_new_concepts_edlora_
        embedding_features, _ = merge_new_concepts_(embedding_list, concept_list, tokenizer, text_encoder)
    else:
        embedding_features, _ = {}, {}
        logger.info('------Step 3: no new embedding, skip merging token embedding------')

    # step 2: construct reparameterized text_encoder
    if any([item is not None for item in text_encoder_list]):
        logger.info('------Step 4: merge text encoder------')
        new_text_encoder_weights = merge_text_encoder(concept_list, text_encoder, text_encoder_list, device)
    else:
        new_text_encoder_weights = {}
        logger.info('------Step 4: no new text encoder, skip merging text encoder------')

    # step 2: construct reparameterized text_encoder
    if any([item is not None for item in unet_list]):
        logger.info('------Step 5: merge unet------')
        new_unet_weights = merge_unet(concept_list, unet, unet_list, device)
    else:
        new_unet_weights = {}
        logger.info('------Step 4: no new unet, skip merging unet------')

    new_weights = {
        'unet': new_unet_weights,
        'text_encoder': new_text_encoder_weights,
        'new_concept_embedding': embedding_features
    }
    os.makedirs(f'{save_path}', exist_ok=True)
    torch.save({'params': new_weights}, f'{save_path}/combined_model_{suffix}.pth')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--concept_cfg', help='json file for multi-concept', required=True, type=str)
    parser.add_argument('--save_path', help='folder name to save optimized weights', required=True, type=str)
    parser.add_argument('--suffix', help='suffix name', default='', type=str)
    parser.add_argument('--pretrained_models', required=True, type=str)
    parser.add_argument('--type', help='model type', default='edlora', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # s1: set logger
    exp_dir = f'{args.save_path}'
    os.makedirs(exp_dir, exist_ok=True)
    log_file = f'{exp_dir}/combined_model_{args.suffix}.log'
    logger = get_root_logger(logger_name='mixofshow', log_level=logging.INFO, log_file=log_file)
    logger.info(args)

    assert args.type in ['lora', 'edlora']

    compose_concepts(args.concept_cfg, args.pretrained_models, args.save_path, args.suffix, args.type, device='cpu')
