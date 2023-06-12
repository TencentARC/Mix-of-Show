combined_model_root="experiments/MixofShow_Results/Fused_Models"
expdir="hina+kario+tezuka+mitsuha+son_anythingv4"

#---------------------------------------------five_anime-------------------------------------------

five_anime=1

if [ ${five_anime} -eq 1 ]
then
  keypose_condition='datasets/validation_spatial_condition/multi-characters/anime_pose/hina_tezuka_mitsuha_goku_kaori.png'
  keypose_adaptor_weight=1.0
  sketch_condition=''
  sketch_adaptor_weight=1.0

  context_prompt='three girls, a boy and a goku are walking near a lake'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <hina1> <hina2>, near a lake]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[61, 18, 512, 192]'

  region2_prompt='[a <tezuka1> <tezuka2>, near a lake]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[20, 194, 512, 407]'

  region3_prompt='[a <mitsuha1> <mitsuha2>, near a lake]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[82, 394, 512, 629]'

  region4_prompt='[a <son1> <son2>, near a lake]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[9, 605, 512, 803]'

  region5_prompt='[a <kaori1> <kaori2>, near a lake]'
  region5_neg_prompt="[${context_neg_prompt}]"
  region5='[71, 801, 512, 978]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}|${region5_prompt}-*-${region5_neg_prompt}-*-${region5}"

  python inference/mix_of_show_sample.py \
    --pretrained_model="experiments/pretrained_models/anything-v4.0" \
    --combined_model="${combined_model_root}/${expdir}/combined_model.pth" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adpator/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adpator/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="results/multi-concept/${expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=14
fi

#---------------------------------------------three_anime_attr-------------------------------------------

three_anime_attr=1

if [ ${three_anime_attr} -eq 1 ]
then
  keypose_condition='datasets/validation_spatial_condition/multi-characters/anime_pose/hina_tezuka_kario.png'
  keypose_adaptor_weight=1.0
  sketch_condition=''
  sketch_adaptor_weight=1.0

  context_prompt='two girls and a boy are walking near a lake'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <hina1> <hina2>, in red dress, near a lake]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[61, 115, 512, 273]'

  region2_prompt='[a <tezuka1> <tezuka2>, in white suit, near a lake]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[19, 292, 512, 512]'

  region3_prompt='[a <kaori1> <kaori2>, wearing a hat, near a lake]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[48, 519, 512, 706]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"


  python inference/mix_of_show_sample.py \
    --pretrained_model="experiments/pretrained_models/anything-v4.0" \
    --combined_model="${combined_model_root}/${expdir}/combined_model.pth" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adpator/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adpator/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="results/multi-concept/${expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=14641
fi

#---------------------------------------------three_anime_girl-------------------------------------------

three_anime_girl=1

if [ ${three_anime_girl} -eq 1 ]
then
  keypose_condition='datasets/validation_spatial_condition/multi-characters/anime_pose/hina_mitsuha_kario.png'
  keypose_adaptor_weight=1.0
  sketch_condition=''
  sketch_adaptor_weight=1.0

  context_prompt='three girls are walking near a lake'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <hina1> <hina2>, near a lake]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[61, 115, 512, 273]'

  region2_prompt='[a <mitsuha1> <mitsuha2>, near a lake]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[49, 323, 512, 500]'

  region3_prompt='[a <kaori1> <kaori2>, near a lake]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[53, 519, 512, 715]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"


  python inference/mix_of_show_sample.py \
    --pretrained_model="experiments/pretrained_models/anything-v4.0" \
    --combined_model="${combined_model_root}/${expdir}/combined_model.pth" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adpator/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adpator/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="results/multi-concept/${expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=14641
fi
