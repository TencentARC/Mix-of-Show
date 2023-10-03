import argparse
import copy
import itertools
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline
from tqdm import tqdm

from mixofshow.models.edlora import revise_edlora_unet_attention_forward
from mixofshow.pipelines.pipeline_edlora import bind_concept_prompt
from mixofshow.utils.util import set_logger

TEMPLATE_SIMPLE = 'photo of a {}'


def chunk_compute_mse(K_target, V_target, W, device, chunk_size=5000):
    num_chunks = (K_target.size(0) + chunk_size - 1) // chunk_size

    loss = 0

    for i in range(num_chunks):
        # Extract the current chunk
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, K_target.size(0))
        loss += F.mse_loss(
            F.linear(K_target[start_idx:end_idx].to(device), W),
            V_target[start_idx:end_idx].to(device)) * (end_idx - start_idx)
    loss /= K_target.size(0)
    return loss


def update_quasi_newton(K_target, V_target, W, iters, device):
    '''
    Args:
        K: torch.Tensor, size [n_samples, n_features]
        V: torch.Tensor, size [n_samples, n_targets]
        K_target: torch.Tensor, size [n_constraints, n_features]
        V_target: torch.Tensor, size [n_constraints, n_targets]
        W: torch.Tensor, size [n_targets, n_features]

    Returns:
        Wnew: torch.Tensor, size [n_targets, n_features]
    '''

    W = W.detach()
    V_target = V_target.detach()
    K_target = K_target.detach()

    W.requires_grad = True
    K_target.requires_grad = False
    V_target.requires_grad = False

    best_loss = np.Inf
    best_W = None

    def closure():
        nonlocal best_W, best_loss
        optimizer.zero_grad()

        if len(W.shape) == 4:
            loss = F.mse_loss(F.conv2d(K_target.to(device), W),
                              V_target.to(device))
        else:
            loss = chunk_compute_mse(K_target, V_target, W, device)

        if loss < best_loss:
            best_loss = loss
            best_W = W.clone().cpu()
        loss.backward()
        return loss

    optimizer = optim.LBFGS([W],
                            lr=1,
                            max_iter=iters,
                            history_size=25,
                            line_search_fn='strong_wolfe',
                            tolerance_grad=1e-16,
                            tolerance_change=1e-16)
    optimizer.step(closure)

    with torch.no_grad():
        if len(W.shape) == 4:
            loss = torch.norm(
                F.conv2d(K_target.to(torch.float32), best_W.to(torch.float32)) - V_target.to(torch.float32), 2, dim=1)
        else:
            loss = torch.norm(
                F.linear(K_target.to(torch.float32), best_W.to(torch.float32)) - V_target.to(torch.float32), 2, dim=1)

    logging.info('new_concept loss: %e' % loss.mean().item())
    return best_W


def merge_lora_into_weight(original_state_dict, lora_state_dict, modification_layer_names, model_type, alpha, device):
    def get_lora_down_name(original_layer_name):
        if model_type == 'text_encoder':
            lora_down_name = original_layer_name.replace('q_proj.weight', 'q_proj.lora_down.weight') \
                .replace('k_proj.weight', 'k_proj.lora_down.weight') \
                .replace('v_proj.weight', 'v_proj.lora_down.weight') \
                .replace('out_proj.weight', 'out_proj.lora_down.weight') \
                .replace('fc1.weight', 'fc1.lora_down.weight') \
                .replace('fc2.weight', 'fc2.lora_down.weight')
        else:
            lora_down_name = k.replace('to_q.weight', 'to_q.lora_down.weight') \
                .replace('to_k.weight', 'to_k.lora_down.weight') \
                .replace('to_v.weight', 'to_v.lora_down.weight') \
                .replace('to_out.0.weight', 'to_out.0.lora_down.weight') \
                .replace('ff.net.0.proj.weight', 'ff.net.0.proj.lora_down.weight') \
                .replace('ff.net.2.weight', 'ff.net.2.lora_down.weight') \
                .replace('proj_out.weight', 'proj_out.lora_down.weight') \
                .replace('proj_in.weight', 'proj_in.lora_down.weight')

        return lora_down_name

    assert model_type in ['unet', 'text_encoder']
    new_state_dict = copy.deepcopy(original_state_dict)
    load_cnt = 0

    for k in modification_layer_names:
        lora_down_name = get_lora_down_name(k)
        lora_up_name = lora_down_name.replace('lora_down', 'lora_up')

        if lora_up_name in lora_state_dict:
            load_cnt += 1
            original_params = new_state_dict[k]
            lora_down_params = lora_state_dict[lora_down_name].to(device)
            lora_up_params = lora_state_dict[lora_up_name].to(device)
            if len(original_params.shape) == 4:
                lora_param = lora_up_params.squeeze(
                ) @ lora_down_params.squeeze()
                lora_param = lora_param.unsqueeze(-1).unsqueeze(-1)
            else:
                lora_param = lora_up_params @ lora_down_params
            merge_params = original_params + alpha * lora_param
            new_state_dict[k] = merge_params

    logging.info(f'load {load_cnt} LoRAs of {model_type}')
    return new_state_dict


module_io_recoder = {}
record_feature = False  # remember to set record feature


def get_hooker(module_name):
    def hook(module, feature_in, feature_out):
        if module_name not in module_io_recoder:
            module_io_recoder[module_name] = {'input': [], 'output': []}
        if record_feature:
            module_io_recoder[module_name]['input'].append(feature_in[0].cpu())
            if module.bias is not None:
                if len(feature_out.shape) == 4:
                    bias = module.bias.unsqueeze(-1).unsqueeze(-1)
                else:
                    bias = module.bias
                module_io_recoder[module_name]['output'].append(
                    (feature_out - bias).cpu())  # remove bias
            else:
                module_io_recoder[module_name]['output'].append(
                    feature_out.cpu())

    return hook


def init_stable_diffusion(pretrained_model_path, device):
    # step1: get w0 parameters
    model_id = pretrained_model_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    train_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder='scheduler')
    test_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe.safety_checker = None
    pipe.scheduler = test_scheduler
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

            new_token_position = torch.where(torch.tensor(tokens) >= 49407)[0]
            # >40497 not include end token | >=40497 include end token
            concept_feature = text_encoder(
                torch.LongTensor(tokens).reshape(
                    1, -1).to(device))[0][:,
                              new_token_position].reshape(-1, 768)
            text_features.append(concept_feature)
        return torch.cat(text_features, 0).float()
    elif return_type == 'full_embedding':
        text_input = tokenizer(prompts,
                               padding='max_length',
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors='pt')
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        return text_embeddings
    else:
        raise NotImplementedError


def merge_new_concepts_(embedding_list, concept_list, tokenizer, text_encoder):
    def add_new_concept(concept_name, embedding):
        new_token_names = [
            f'<new{start_idx + layer_id}>'
            for layer_id in range(NUM_CROSS_ATTENTION_LAYERS)
        ]
        num_added_tokens = tokenizer.add_tokens(new_token_names)
        assert num_added_tokens == NUM_CROSS_ATTENTION_LAYERS
        new_token_ids = [
            tokenizer.convert_tokens_to_ids(token_name)
            for token_name in new_token_names
        ]

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data

        token_embeds[new_token_ids] = token_embeds[new_token_ids].copy_(
            embedding[concept_name])

        embedding_features.update({concept_name: embedding[concept_name]})
        logging.info(
            f'concept {concept_name} is bind with token_id: [{min(new_token_ids)}, {max(new_token_ids)}]'
        )

        return start_idx + NUM_CROSS_ATTENTION_LAYERS, new_token_ids, new_token_names

    embedding_features = {}
    new_concept_cfg = {}

    start_idx = 0

    NUM_CROSS_ATTENTION_LAYERS = 16

    for idx, (embedding,
              concept) in enumerate(zip(embedding_list, concept_list)):
        concept_names = concept['concept_name'].split(' ')

        for concept_name in concept_names:
            if not concept_name.startswith('<'):
                continue
            else:
                assert concept_name in embedding, 'check the config, the provide concept name is not in the lora model'
            start_idx, new_token_ids, new_token_names = add_new_concept(
                concept_name, embedding)
            new_concept_cfg.update({
                concept_name: {
                    'concept_token_ids': new_token_ids,
                    'concept_token_names': new_token_names
                }
            })
    return embedding_features, new_concept_cfg


def parse_new_concepts(concept_cfg):
    with open(concept_cfg, 'r') as f:
        concept_list = json.load(f)

    model_paths = [concept['lora_path'] for concept in concept_list]

    embedding_list = []
    text_encoder_list = []
    unet_crosskv_list = []
    unet_spatial_attn_list = []

    for model_path in model_paths:
        model = torch.load(model_path)['params']

        if 'new_concept_embedding' in model and len(
                model['new_concept_embedding']) != 0:
            embedding_list.append(model['new_concept_embedding'])
        else:
            embedding_list.append(None)

        if 'text_encoder' in model and len(model['text_encoder']) != 0:
            text_encoder_list.append(model['text_encoder'])
        else:
            text_encoder_list.append(None)

        if 'unet' in model and len(model['unet']) != 0:
            crosskv_matches = ['attn2.to_k.lora', 'attn2.to_v.lora']
            crosskv_dict = {
                k: v
                for k, v in model['unet'].items()
                if any([x in k for x in crosskv_matches])
            }

            if len(crosskv_dict) != 0:
                unet_crosskv_list.append(crosskv_dict)
            else:
                unet_crosskv_list.append(None)

            spatial_attn_dict = {
                k: v
                for k, v in model['unet'].items()
                if all([x not in k for x in crosskv_matches])
            }

            if len(spatial_attn_dict) != 0:
                unet_spatial_attn_list.append(spatial_attn_dict)
            else:
                unet_spatial_attn_list.append(None)
        else:
            unet_crosskv_list.append(None)
            unet_spatial_attn_list.append(None)

    return embedding_list, text_encoder_list, unet_crosskv_list, unet_spatial_attn_list, concept_list


def merge_kv_in_cross_attention(concept_list, optimize_iters, new_concept_cfg,
                                tokenizer, text_encoder, unet,
                                unet_crosskv_list, device):
    # crosskv attention layer names
    matches = ['attn2.to_k', 'attn2.to_v']

    cross_attention_idx = -1
    cross_kv_layer_names = []

    # the crosskv name should match the order down->mid->up, and record its layer id
    for name, _ in unet.down_blocks.named_parameters():
        if any([x in name for x in matches]):
            if 'to_k' in name:
                cross_attention_idx += 1
                cross_kv_layer_names.append(
                    (cross_attention_idx, 'down_blocks.' + name))
                cross_kv_layer_names.append(
                    (cross_attention_idx,
                     'down_blocks.' + name.replace('to_k', 'to_v')))
            else:
                pass

    for name, _ in unet.mid_block.named_parameters():
        if any([x in name for x in matches]):
            if 'to_k' in name:
                cross_attention_idx += 1
                cross_kv_layer_names.append(
                    (cross_attention_idx, 'mid_block.' + name))
                cross_kv_layer_names.append(
                    (cross_attention_idx,
                     'mid_block.' + name.replace('to_k', 'to_v')))
            else:
                pass

    for name, _ in unet.up_blocks.named_parameters():
        if any([x in name for x in matches]):
            if 'to_k' in name:
                cross_attention_idx += 1
                cross_kv_layer_names.append(
                    (cross_attention_idx, 'up_blocks.' + name))
                cross_kv_layer_names.append(
                    (cross_attention_idx,
                     'up_blocks.' + name.replace('to_k', 'to_v')))
            else:
                pass

    logging.info(
        f'Unet have {len(cross_kv_layer_names)} linear layer (related to text feature) need to optimize'
    )

    original_unet_state_dict = unet.state_dict()  # original state dict

    new_concept_input_dict = {}
    new_concept_output_dict = {}

    # step 1: construct prompts for new concept -> extract input/target features
    for concept, tuned_state_dict in zip(concept_list, unet_crosskv_list):
        concept_prompt = [
            TEMPLATE_SIMPLE.format(concept['concept_name']),
            concept['concept_name']
        ]
        concept_prompt = bind_concept_prompt(concept_prompt, new_concept_cfg)

        n = len(concept_prompt) // 16
        layer_prompts = [
            tuple(concept_prompt[j * 16 + i] for j in range(n))
            for i in range(16)
        ]

        for layer_idx, layer_name in cross_kv_layer_names:

            # merge params
            original_params = original_unet_state_dict[layer_name]

            # hard coded here: in unet, self/crosskv attention disable bias parameter
            lora_down_name = layer_name.replace('to_k.weight', 'to_k.lora_down.weight').replace('to_v.weight', 'to_v.lora_down.weight')
            lora_up_name = lora_down_name.replace('lora_down', 'lora_up')

            alpha = concept['unet_alpha']

            lora_down_params = tuned_state_dict[lora_down_name].to(device)
            lora_up_params = tuned_state_dict[lora_up_name].to(device)

            merge_params = original_params + alpha * lora_up_params @ lora_down_params

            layer_concept_prompt = list(layer_prompts[layer_idx])

            prompt_feature = get_text_feature(
                layer_concept_prompt,
                tokenizer,
                text_encoder,
                device,
                return_type='category_embedding').cpu()

            if layer_name not in new_concept_input_dict:
                new_concept_input_dict[layer_name] = []

            if layer_name not in new_concept_output_dict:
                new_concept_output_dict[layer_name] = []

            # print(merge_params.shape, prompt_feature.shape)
            # torch.Size([320, 768]) torch.Size([6, 768])
            new_concept_input_dict[layer_name].append(prompt_feature)
            new_concept_output_dict[layer_name].append(
                (merge_params.cpu() @ prompt_feature.T).T)

    for k, v in new_concept_input_dict.items():
        new_concept_input_dict[k] = torch.cat(v, 0)  # torch.Size([14, 768])

    for k, v in new_concept_output_dict.items():
        new_concept_output_dict[k] = torch.cat(v, 0)  # torch.Size([14, 768])

    new_kv_weights = {}
    # step 3: begin update model
    for idx, (layer_idx, layer_name) in enumerate(cross_kv_layer_names):
        W = original_unet_state_dict[layer_name].to(torch.float32)  # origin params

        new_concept_input = new_concept_input_dict[layer_name]
        new_concept_output = new_concept_output_dict[layer_name]

        logging.info(
            f'[{(idx + 1)}/{len(cross_kv_layer_names)}] optimizing {layer_name}')

        Wnew = update_quasi_newton(
            new_concept_input.to(W.dtype),  # our concept
            new_concept_output.to(W.dtype),  # our concept
            W.clone(),
            iters=optimize_iters,
            device=device)

        new_kv_weights[layer_name] = Wnew

    return new_kv_weights


def merge_text_encoder(concept_list, optimize_iters, new_concept_cfg,
                       tokenizer, text_encoder, text_encoder_list, device):
    def process_extract_features(input_feature_list, output_feature_list):
        text_input_features = [
            feat.reshape(-1, feat.shape[-1]) for feat in input_feature_list
        ]
        text_output_features = [
            feat.reshape(-1, feat.shape[-1]) for feat in output_feature_list
        ]
        text_input_features = torch.cat(text_input_features, 0)
        text_output_features = torch.cat(text_output_features, 0)
        return text_input_features, text_output_features

    LoRA_keys = []
    for textenc_lora in text_encoder_list:
        LoRA_keys += list(textenc_lora.keys())
    LoRA_keys = set([
        key.replace('.lora_down', '').replace('.lora_up', '')
        for key in LoRA_keys
    ])
    text_encoder_layer_names = LoRA_keys

    candidate_module_name = [
        'q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'
    ]
    candidate_module_name = [
        name for name in candidate_module_name
        if any([name in key for key in LoRA_keys])
    ]

    logging.info(f'text_encoder have {len(text_encoder_layer_names)} linear layer need to optimize')

    global module_io_recoder, record_feature
    hooker_handlers = []
    for name, module in text_encoder.named_modules():
        if any([item in name for item in candidate_module_name]):
            hooker_handlers.append(module.register_forward_hook(hook=get_hooker(name)))

    logging.info(f'add {len(hooker_handlers)} hooker to text_encoder')

    original_state_dict = copy.deepcopy(text_encoder.state_dict())  # original state dict

    new_concept_input_dict = {}
    new_concept_output_dict = {}

    for concept, lora_state_dict in zip(concept_list, text_encoder_list):

        merged_state_dict = merge_lora_into_weight(
            original_state_dict,
            lora_state_dict,
            text_encoder_layer_names,
            model_type='text_encoder',
            alpha=concept['text_encoder_alpha'],
            device=device)
        text_encoder.load_state_dict(merged_state_dict)  # load merged parameters

        concept_prompt = [
            TEMPLATE_SIMPLE.format(concept['concept_name']),
            concept['concept_name']
        ]
        concept_prompt = bind_concept_prompt(concept_prompt, new_concept_cfg)

        # reinit module io recorder
        module_io_recoder = {}
        record_feature = True
        _ = get_text_feature(concept_prompt,
                             tokenizer,
                             text_encoder,
                             device,
                             return_type='category_embedding')

        # we use different model to compute new concept feature
        for layer_name in text_encoder_layer_names:
            input_feature_list = module_io_recoder[layer_name.replace('.weight', '')]['input']
            output_feature_list = module_io_recoder[layer_name.replace('.weight', '')]['output']

            text_input_features, text_output_features = \
                process_extract_features(input_feature_list, output_feature_list)

            if layer_name not in new_concept_output_dict:
                new_concept_input_dict[layer_name] = []
                new_concept_output_dict[layer_name] = []

            new_concept_input_dict[layer_name].append(text_input_features)
            new_concept_output_dict[layer_name].append(text_output_features)

    for k, v in new_concept_input_dict.items():
        new_concept_input_dict[k] = torch.cat(v, 0)  # torch.Size([14, 768])

    for k, v in new_concept_output_dict.items():
        new_concept_output_dict[k] = torch.cat(v, 0)  # torch.Size([14, 768])

    new_text_encoder_weights = {}
    # step 3: begin update model
    for idx, layer_name in enumerate(text_encoder_layer_names):
        W = original_state_dict[layer_name].to(torch.float32)

        new_concept_input = new_concept_input_dict[layer_name]
        new_concept_target = new_concept_output_dict[layer_name]

        logging.info(f'[{(idx + 1)}/{len(text_encoder_layer_names)}] optimizing {layer_name}')

        Wnew = update_quasi_newton(
            new_concept_input.to(W.dtype),  # our concept
            new_concept_target.to(W.dtype),  # our concept
            W.clone(),
            iters=optimize_iters,
            device=device)
        new_text_encoder_weights[layer_name] = Wnew

    logging.info(f'remove {len(hooker_handlers)} hooker from text_encoder')

    # remove forward hooker
    for hook_handle in hooker_handlers:
        hook_handle.remove()

    return new_text_encoder_weights


@torch.no_grad()
def decode_to_latents(concept_prompt, new_concept_cfg, tokenizer, text_encoder,
                      unet, test_scheduler, num_inference_steps, device,
                      record_nums, batch_size):

    concept_prompt = bind_concept_prompt([concept_prompt], new_concept_cfg)
    text_embeddings = get_text_feature(
        concept_prompt,
        tokenizer,
        text_encoder,
        device,
        return_type='full_embedding').unsqueeze(0)

    text_embeddings = text_embeddings.repeat((batch_size, 1, 1, 1))

    # sd 1.x
    height = 512
    width = 512

    latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8), )
    latents = latents.to(device, dtype=text_embeddings.dtype)

    test_scheduler.set_timesteps(num_inference_steps)
    latents = latents * test_scheduler.init_noise_sigma

    global record_feature
    step = (test_scheduler.timesteps.size(0)) // record_nums
    record_timestep = test_scheduler.timesteps[torch.arange(0, test_scheduler.timesteps.size(0), step=step)[:record_nums]]

    for t in tqdm(test_scheduler.timesteps):

        if t in record_timestep:
            record_feature = True
        else:
            record_feature = False

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = latents
        latent_model_input = test_scheduler.scale_model_input(latent_model_input, t)

        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # compute the previous noisy sample x_t -> x_t-1
        latents = test_scheduler.step(noise_pred, t, latents).prev_sample

    return latents, text_embeddings


def merge_spatial_attention(concept_list, optimize_iters, new_concept_cfg, tokenizer, text_encoder, unet, unet_spatial_attn_list, test_scheduler, device):
    LoRA_keys = []
    for unet_lora in unet_spatial_attn_list:
        LoRA_keys += list(unet_lora.keys())
    LoRA_keys = set([
        key.replace('.lora_down', '').replace('.lora_up', '')
        for key in LoRA_keys
    ])
    spatial_attention_layer_names = LoRA_keys

    candidate_module_name = [
        'attn2.to_q', 'attn2.to_out.0', 'attn1.to_q', 'attn1.to_k',
        'attn1.to_v', 'attn1.to_out.0', 'ff.net.2', 'ff.net.0.proj',
        'proj_out', 'proj_in'
    ]
    candidate_module_name = [
        name for name in candidate_module_name
        if any([name in key for key in LoRA_keys])
    ]

    logging.info(
        f'unet have {len(spatial_attention_layer_names)} linear layer need to optimize'
    )

    global module_io_recoder
    hooker_handlers = []
    for name, module in unet.named_modules():
        if any([x in name for x in candidate_module_name]):
            hooker_handlers.append(
                module.register_forward_hook(hook=get_hooker(name)))

    logging.info(f'add {len(hooker_handlers)} hooker to unet')

    original_state_dict = copy.deepcopy(unet.state_dict())  # original state dict
    revise_edlora_unet_attention_forward(unet)

    new_concept_input_dict = {}
    new_concept_output_dict = {}

    for concept, tuned_state_dict in zip(concept_list, unet_spatial_attn_list):
        # set unet
        module_io_recoder = {}  # reinit module io recorder

        merged_state_dict = merge_lora_into_weight(
            original_state_dict,
            tuned_state_dict,
            spatial_attention_layer_names,
            model_type='unet',
            alpha=concept['unet_alpha'],
            device=device)
        unet.load_state_dict(merged_state_dict)  # load merged parameters

        concept_name = concept['concept_name']
        concept_prompt = TEMPLATE_SIMPLE.format(concept_name)

        decode_to_latents(concept_prompt,
                          new_concept_cfg,
                          tokenizer,
                          text_encoder,
                          unet,
                          test_scheduler,
                          num_inference_steps=20,
                          device=device,
                          record_nums=20,
                          batch_size=1)
        # record record_num * batch size feature for one concept

        for layer_name in spatial_attention_layer_names:
            input_feature_list = module_io_recoder[layer_name.replace('.weight', '')]['input']
            output_feature_list = module_io_recoder[layer_name.replace('.weight', '')]['output']

            text_input_features, text_output_features = \
                torch.cat(input_feature_list, 0), torch.cat(output_feature_list, 0)

            if layer_name not in new_concept_output_dict:
                new_concept_input_dict[layer_name] = []
                new_concept_output_dict[layer_name] = []

            new_concept_input_dict[layer_name].append(text_input_features)
            new_concept_output_dict[layer_name].append(text_output_features)

    for k, v in new_concept_input_dict.items():
        new_concept_input_dict[k] = torch.cat(v, 0)

    for k, v in new_concept_output_dict.items():
        new_concept_output_dict[k] = torch.cat(v, 0)

    new_spatial_attention_weights = {}

    # step 5: begin update model
    for idx, layer_name in enumerate(spatial_attention_layer_names):
        new_concept_input = new_concept_input_dict[layer_name]
        if len(new_concept_input.shape) == 4:
            new_concept_input = new_concept_input
        else:
            new_concept_input = new_concept_input.reshape(-1, new_concept_input.shape[-1])

        new_concept_output = new_concept_output_dict[layer_name]
        if len(new_concept_output.shape) == 4:
            new_concept_output = new_concept_output
        else:
            new_concept_output = new_concept_output.reshape(-1, new_concept_output.shape[-1])

        logging.info(f'[{(idx + 1)}/{len(spatial_attention_layer_names)}] optimizing {layer_name}')

        W = original_state_dict[layer_name].to(torch.float32)  # origin params
        Wnew = update_quasi_newton(
            new_concept_input.to(W.dtype),  # our concept
            new_concept_output.to(W.dtype),  # our concept
            W.clone(),
            iters=optimize_iters,
            device=device,
        )
        new_spatial_attention_weights[layer_name] = Wnew

    logging.info(f'remove {len(hooker_handlers)} hooker from unet')

    for hook_handle in hooker_handlers:
        hook_handle.remove()

    return new_spatial_attention_weights


def compose_concepts(concept_cfg, optimize_textenc_iters, optimize_unet_iters, pretrained_model_path, save_path, suffix, device):
    logging.info('------Step 1: load stable diffusion checkpoint------')
    pipe, train_scheduler, test_scheduler = init_stable_diffusion(pretrained_model_path, device)
    tokenizer, text_encoder, unet, vae = pipe.tokenizer, pipe.text_encoder, pipe.unet, pipe.vae
    for param in itertools.chain(text_encoder.parameters(), unet.parameters(), vae.parameters()):
        param.requires_grad = False

    logging.info('------Step 2: load new concepts checkpoints------')
    embedding_list, text_encoder_list, unet_crosskv_list, unet_spatial_attn_list, concept_list = parse_new_concepts(concept_cfg)

    # step 1: inplace add new concept to tokenizer and embedding layers of text encoder
    if any([item is not None for item in embedding_list]):
        logging.info('------Step 3: merge token embedding------')
        _, new_concept_cfg = merge_new_concepts_(embedding_list, concept_list, tokenizer, text_encoder)
    else:
        _, new_concept_cfg = {}, {}
        logging.info('------Step 3: no new embedding, skip merging token embedding------')

    # step 2: construct reparameterized text_encoder
    if any([item is not None for item in text_encoder_list]):
        logging.info('------Step 4: merge text encoder------')
        new_text_encoder_weights = merge_text_encoder(
            concept_list, optimize_textenc_iters, new_concept_cfg, tokenizer,
            text_encoder, text_encoder_list, device)
        # update the merged state_dict in text_encoder
        text_encoder_state_dict = text_encoder.state_dict()
        text_encoder_state_dict.update(new_text_encoder_weights)
        text_encoder.load_state_dict(text_encoder_state_dict)
    else:
        new_text_encoder_weights = {}
        logging.info('------Step 4: no new text encoder, skip merging text encoder------')

    # step 3: merge unet (k,v in crosskv-attention) params, since they only receive input from text-encoder

    if any([item is not None for item in unet_crosskv_list]):
        logging.info('------Step 5: merge kv of cross-attention in unet------')
        new_kv_weights = merge_kv_in_cross_attention(
            concept_list, optimize_textenc_iters, new_concept_cfg,
            tokenizer, text_encoder, unet, unet_crosskv_list, device)
        # update the merged state_dict in kv of crosskv-attention in Unet
        unet_state_dict = unet.state_dict()
        unet_state_dict.update(new_kv_weights)
        unet.load_state_dict(unet_state_dict)
    else:
        new_kv_weights = {}
        logging.info('------Step 5: no new kv of cross-attention in unet, skip merging kv------')

    # step 4: merge unet (q,k,v in self-attention, q in crosskv-attention)
    if any([item is not None for item in unet_spatial_attn_list]):
        logging.info('------Step 6: merge spatial attention (q in cross-attention, qkv in self-attention) in unet------')
        new_spatial_attention_weights = merge_spatial_attention(
            concept_list, optimize_unet_iters, new_concept_cfg, tokenizer,
            text_encoder, unet, unet_spatial_attn_list, test_scheduler, device)
        unet_state_dict = unet.state_dict()
        unet_state_dict.update(new_spatial_attention_weights)
        unet.load_state_dict(unet_state_dict)
    else:
        new_spatial_attention_weights = {}
        logging.info('------Step 6: no new spatial-attention in unet, skip merging spatial attention------')

    checkpoint_save_path = f'{save_path}/combined_model_{suffix}'
    pipe.save_pretrained(checkpoint_save_path)
    with open(os.path.join(checkpoint_save_path, 'new_concept_cfg.json'), 'w') as json_file:
        json.dump(new_concept_cfg, json_file)


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--concept_cfg', help='json file for multi-concept', required=True, type=str)
    parser.add_argument('--save_path', help='folder name to save optimized weights', required=True, type=str)
    parser.add_argument('--suffix', help='suffix name', default='base', type=str)
    parser.add_argument('--pretrained_models', required=True, type=str)
    parser.add_argument('--optimize_unet_iters', default=50, type=int)
    parser.add_argument('--optimize_textenc_iters', default=500, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # s1: set logger
    exp_dir = f'{args.save_path}'
    os.makedirs(exp_dir, exist_ok=True)
    log_file = f'{exp_dir}/combined_model_{args.suffix}.log'
    set_logger(log_file=log_file)
    logging.info(args)

    compose_concepts(args.concept_cfg,
                     args.optimize_textenc_iters,
                     args.optimize_unet_iters,
                     args.pretrained_models,
                     args.save_path,
                     args.suffix,
                     device='cuda')
