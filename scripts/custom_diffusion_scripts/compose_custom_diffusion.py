import argparse
import itertools
import json
import logging
import os
import os.path as osp
import torch
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from scipy.linalg import lu_factor, lu_solve

from mixofshow.utils import get_root_logger, mkdir_and_rename


def gdupdateWexact(K, V, K_target, V_target, W):

    # because of some cuda version (11.3) have bugs, we move to cpu do this calculation
    original_device = W.device
    W = W.cpu()
    K = K.cpu()
    V = V.cpu()
    V_target = V_target.cpu()
    K_target = K_target.cpu()

    input_ = K
    output = V
    C = input_.T @ input_
    d = []
    lu, piv = lu_factor(C.cpu().numpy())
    for i in range(K_target.size(0)):
        sol = lu_solve((lu, piv), K_target[i].reshape(-1, 1).cpu().numpy())
        d.append(torch.from_numpy(sol).to(K.device))

    d = torch.cat(d, 1).T

    e2 = d @ K_target.T
    e1 = (V_target.T - W @ K_target.T)
    delta = e1 @ torch.linalg.inv(e2)

    Wnew = W + delta @ d
    lambda_split1 = V_target.size(0)

    input_ = torch.cat([K_target.T, K.T], dim=1)
    output = torch.cat([V_target, V], dim=0)
    loss = torch.norm((Wnew @ input_).T - output, 2, dim=1)
    logger.info('new_concept loss: %e, old_sd loss: %e' %
                (loss[:lambda_split1].mean().item(), loss[lambda_split1:].mean().item()))

    return Wnew.to(original_device)


def init_stable_diffusion(pretrained_model_path, device):
    # step1: get w0 parameters
    model_id = pretrained_model_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    train_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder='scheduler')
    test_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
    return pipe, train_scheduler, test_scheduler


@torch.no_grad()
def get_text_feature(prompts, tokenizer, text_encoder, device, return_type='category_embedding'):
    text_features = []

    if return_type == 'category_embedding':
        for text in prompts:

            tokens = tokenizer(
                text,
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_length=True,
                return_overflowing_tokens=False,
                padding='do_not_pad',
            ).input_ids

            if 'photo of a' in text[:15]:
                # no consider same token
                text_features.append(
                    text_encoder(torch.LongTensor(tokens).reshape(1, -1).to(device))[0][:, 4:].reshape(-1, 768))
            else:
                text_features.append(
                    text_encoder(torch.LongTensor(tokens).reshape(1, -1).to(device))[0][:, 1:].reshape(-1, 768))
        return torch.cat(text_features, 0).float()
    elif return_type == 'full_embedding':
        text_input = tokenizer(
            prompts, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        return text_embeddings
    else:
        raise NotImplementedError


def merge_new_concepts_(embedding_list, concept_list, tokenizer, text_encoder):
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
    return embedding_features


def parse_new_concepts(concept_cfg):
    with open(concept_cfg, 'r') as f:
        concept_list = json.load(f)

    model_paths = [concept['lora_path'] for concept in concept_list]

    embedding_list = []
    unet_crosskv_list = []

    for model_path in model_paths:
        model = torch.load(model_path)['params']

        if 'modifier_token' in model and len(model['modifier_token']) != 0:
            embedding_list.append(model['modifier_token'])
        else:
            embedding_list.append(None)

        if 'unet' in model and len(model['unet']) != 0:
            crosskv_matches = ['attn2.to_k', 'attn2.to_v']
            crosskv_dict = {k: v for k, v in model['unet'].items() if any([x in k for x in crosskv_matches])}

            if len(crosskv_dict) != 0:
                unet_crosskv_list.append(crosskv_dict)
            else:
                unet_crosskv_list.append(None)

    return embedding_list, unet_crosskv_list, concept_list


def merge_kv_in_cross_attention(regularization_root, concept_list, tokenizer, text_encoder, unet, unet_crosskv_list,
                                device):
    logger = get_root_logger()

    # crosskv attention layer names
    matches = ['attn2.to_k', 'attn2.to_v']
    cross_kv_layer_names = [name for name, _ in unet.named_parameters() if any([x in name for x in matches])]

    logger.info(f'Unet have {len(cross_kv_layer_names)} linear layer (related to text feature) need to optimize')

    original_unet_state_dict = unet.state_dict()  # original state dict

    new_concept_target_dict = {}

    # step 1: construct prompts for new concept -> extract input/target features
    new_concept_text_feature = []
    for concept, tuned_state_dict in zip(concept_list, unet_crosskv_list):
        string = concept['concept_name']

        prompt = [string] + [f'photo of a {string}']  # ["<new1> cat", "photo of a <new1> cat"]

        prompt_feature = get_text_feature(prompt, tokenizer, text_encoder, device, return_type='category_embedding')
        # print(prompt_feature.shape)  # torch.Size([6, 768])

        new_concept_text_feature.append(prompt_feature)

        # we use different model to compute new concept feature
        for layer_name in cross_kv_layer_names:
            # merge params
            new_params = tuned_state_dict[layer_name]

            if layer_name not in new_concept_target_dict:
                new_concept_target_dict[layer_name] = []
            # print(merge_params.shape, prompt_feature.shape)
            # torch.Size([320, 768]) torch.Size([6, 768])
            new_concept_target_dict[layer_name].append((new_params.to(device) @ prompt_feature.T).T)

    new_concept_text_feature = torch.cat(new_concept_text_feature, 0)  # torch.Size([14, 768])

    for k, v in new_concept_target_dict.items():
        new_concept_target_dict[k] = torch.cat(v, 0)  # torch.Size([14, 320])

    # step 2: extract input feature from regularization prompts
    with open(osp.join(regularization_root, 'caption.txt'), 'r') as fr:
        lines = fr.readlines()
        lines = [line.strip() for line in lines][:200]

    reg_text_feature = get_text_feature(lines, tokenizer, text_encoder, device, return_type='category_embedding')
    # print(reg_prompt_feature.shape) # torch.Size([2883, 768]) combine all sentence of regularization dataset

    new_kv_weights = {}
    # step 3: begin update model
    for idx, layer_name in enumerate(cross_kv_layer_names):
        W = original_unet_state_dict[layer_name]  # origin params
        reg_text_target = (W @ reg_text_feature.T).T

        new_concept_input = new_concept_text_feature
        new_concept_target = new_concept_target_dict[layer_name]

        logger.info(f'[{(idx+1)}/{len(cross_kv_layer_names)}] optimizing {layer_name}')

        Wnew = gdupdateWexact(
            reg_text_feature[:reg_text_target.shape[0]],  # reg input
            reg_text_target,  # reg output
            new_concept_input,  # our concept
            new_concept_target,  # our concept
            W.clone(),
        )

        new_kv_weights[layer_name] = Wnew

    return new_kv_weights


def compose_concepts(concept_cfg, pretrained_model_path, regularization_root, save_path, suffix, device):
    logger = get_root_logger()

    logger.info('------Step 1: load stable diffusion checkpoint------')
    pipe, train_scheduler, test_scheduler = init_stable_diffusion(pretrained_model_path, device)
    tokenizer, text_encoder, unet, vae = pipe.tokenizer, pipe.text_encoder, pipe.unet, pipe.vae
    for param in itertools.chain(text_encoder.parameters(), unet.parameters(), vae.parameters()):
        param.requires_grad = False

    logger.info('------Step 2: load new concepts checkpoints------')
    embedding_list, unet_crosskv_list, concept_list = parse_new_concepts(concept_cfg)

    # step 1: inplace add new concept to tokenizer and embedding layers of text encoder
    if any([item is not None for item in embedding_list]):
        logger.info('------Step 3: merge token embedding------')
        embedding_features = merge_new_concepts_(embedding_list, concept_list, tokenizer, text_encoder)
    else:
        embedding_features = {}
        logger.info('------Step 3: no new embedding, skip merging token embedding------')

    # step 3: merge unet (k,v in crosskv-attention) params, since they only receive input from text-encoder
    if any([item is not None for item in unet_crosskv_list]):
        logger.info('------Step 5: merge kv of cross-attention in unet------')
        new_kv_weights = merge_kv_in_cross_attention(regularization_root, concept_list, tokenizer, text_encoder, unet,
                                                     unet_crosskv_list, device)
    else:
        new_kv_weights = {}
        logger.info('------Step 5: no new kv of cross-attention in unet, skip merging kv------')

    new_weights = {'unet': new_kv_weights, 'new_concept_embedding': embedding_features}
    os.makedirs(f'{save_path}', exist_ok=True)
    torch.save({'params': new_weights}, f'{save_path}/combined_model_{suffix}.pth')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--concept_cfg', help='json file for multi-concept', required=True, type=str)
    parser.add_argument('--save_path', help='folder name to save optimized weights', required=True, type=str)
    parser.add_argument('--suffix', help='suffix name', default='', type=str)
    parser.add_argument('--pretrained_models', default='experiments/pretrained_models/chilloutmix', type=str)
    parser.add_argument('--regularization_root', default='', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # s1: set logger
    exp_dir = f'{args.save_path}'
    mkdir_and_rename(exp_dir)
    log_file = f'{exp_dir}/train.log'
    logger = get_root_logger(logger_name='mixofshow', log_level=logging.INFO, log_file=log_file)
    logger.info(args)

    compose_concepts(
        args.concept_cfg, args.pretrained_models, args.regularization_root, args.save_path, args.suffix, device='cpu')
