# fuse real character
config_file="potter+hermione+thanos_chilloutmix"

python gradient_fusion.py \
    --concept_cfg="datasets/data_cfgs/MixofShow/multi-concept/real/${config_file}.json" \
    --save_path="experiments/composed_edlora/chilloutmix/${config_file}" \
    --pretrained_models="experiments/pretrained_models/chilloutmix" \
    --optimize_textenc_iters=500 \
    --optimize_unet_iters=50

# fuse anime character
config_file="hina+kario+tezuka_anythingv4"

python gradient_fusion.py \
    --concept_cfg="datasets/data_cfgs/MixofShow/multi-concept/anime/${config_file}.json" \
    --save_path="experiments/composed_edlora/anythingv4/${config_file}" \
    --pretrained_models="experiments/pretrained_models/anything-v4.0" \
    --optimize_textenc_iters=500 \
    --optimize_unet_iters=50
