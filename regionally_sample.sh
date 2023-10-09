#---------------------------------------------anime-------------------------------------------

anime_character=0

if [ ${anime_character} -eq 1 ]
then
  fused_model="experiments/composed_edlora/anythingv4/hina+kario+tezuka_anythingv4/combined_model_base"
  expdir="hina+kario+tezuka_anythingv4"

  keypose_condition='datasets/validation_spatial_condition/multi-characters/anime_pose_2x/hina_tezuka_kario_2x.png'
  keypose_adaptor_weight=1.0
  sketch_condition=''
  sketch_adaptor_weight=1.0

  context_prompt='two girls and a boy are standing near a forest'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <hina1> <hina2>, standing near a forest]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[12, 36, 1024, 600]'

  region2_prompt='[a <tezuka1> <tezuka2>, standing near a forest]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[18, 696, 1024, 1180]'

  region5_prompt='[a <kaori1> <kaori2>, standing near a forest]'
  region5_neg_prompt="[${context_neg_prompt}]"
  region5='[142, 1259, 1024, 1956]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region5_prompt}-*-${region5_neg_prompt}-*-${region5}"

  python regionally_controlable_sampling.py \
    --pretrained_model=${fused_model} \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="results/multi-concept/${expdir}" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=19
fi

#---------------------------------------------real-------------------------------------------

real_character=1

if [ ${real_character} -eq 1 ]
then
  fused_model="experiments/composed_edlora/chilloutmix/potter+hermione+thanos_chilloutmix/combined_model_base"
  expdir="potter+hermione+thanos_chilloutmix"

  keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose_2x/harry_hermione_thanos_2x.png'
  keypose_adaptor_weight=1.0

  sketch_condition=''
  sketch_adaptor_weight=1.0

  context_prompt='three people near the castle, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <potter1> <potter2>, in Hogwarts uniform, holding hands, near the castle, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[4, 6, 1024, 490]'

  region2_prompt='[a <hermione1> <hermione2>, girl, in Hogwarts uniform, near the castle, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[14, 490, 1024, 920]'

  region3_prompt='[a <thanos1> <thanos2>, purple armor, near the castle, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[2, 1302, 1024, 1992]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"

  python regionally_controlable_sampling.py \
    --pretrained_model=${fused_model} \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="results/multi-concept/${expdir}" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=14
fi
