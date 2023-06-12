combined_model_root="experiments/MixofShow_Results/Fused_Models"
expdir="potter+hermione+thanos+hinton+lecun+bengio+catA+dogA+chair+table+dogB+vase+pyramid+rock_chilloutmix"

#---------------------------------------------sample people_chair_table_scene-------------------------------------------
people_chair_table_scene=1

if [ ${people_chair_table_scene} -eq 1 ]
then
  keypose_condition='datasets/validation_spatial_condition/characters-objects/bengio+lecun+chair_pose.png'
  keypose_adaptor_weight=0.8

  sketch_condition='datasets/validation_spatial_condition/characters-objects/bengio+lecun+chair_sketch.png'
  sketch_adaptor_weight=0.5

  context_prompt='two men, two chairs, and a table, at the <pyramid1> <pyramid2>, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[<lecun1> <lecun2> sit on a <chair1> <chair2>, wearing a suit, at the <pyramid1> <pyramid2>, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[31, 2, 512, 376]'

  region2_prompt='[serious <bengio1> <bengio2> sit on a <chair1> <chair2>, wearing a suit, at the <pyramid1> <pyramid2>, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[smile, ${context_neg_prompt}]"
  region2='[30, 644, 506, 1011]'

  region3_prompt='[a <vase1> <vase2>, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[110, 438, 342, 568]'

  region4_prompt='[a <table1> <table2>, at the <pyramid1> <pyramid2>, 4K, high quality, high resolution, best quality]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[290, 344, 488, 674]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}"

  python inference/mix_of_show_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
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
    --seed=641
fi

#---------------------------------------------samoke potter_rock---------------------------------------------
potter_rock=1

if [ ${potter_rock} -eq 1 ]
then
  context_prompt='a man and a woman, a cat and a dog, at <rock1> <rock2>, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[<potter1> <potter2>, in hogwarts school uniform, holding hands, at <rock1> <rock2>, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[0, 315, 512, 530]'

  region2_prompt='[<hermione1> <hermione2>, in hogwarts school uniform, holding hands, at <rock1> <rock2>, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[0, 502, 512, 747]'

  region3_prompt='[<dogA1> <dogA2>, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[221, 43, 512, 258]'

  region4_prompt='[<catA1> <catA2>, 4K, high quality, high resolution, best quality]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[228, 752, 512, 1016]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}"

  keypose_condition='datasets/validation_spatial_condition/characters-objects/harry_heminone_scene_pose.png'
  keypose_adaptor_weight=1.0
  region_keypose_adaptor_weight=""

  sketch_condition='datasets/validation_spatial_condition/characters-objects/harry_heminone_scene_sketch.png'
  sketch_adaptor_weight=0.5
  region_sketch_adaptor_weight="${region3}-0.8|${region4}-0.8"

  python inference/mix_of_show_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
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
    --seed=641
fi

#---------------------------------------------sample potter_chair_cat-------------------------------------------
potter_chair_cat=1

if [ ${potter_chair_cat} -eq 1 ]
then
  context_prompt='a man sit in a chair, a dog and a cat sit on a table, in a living room, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[<potter1> <potter2> sit on a <chair1> <chair2>, in a living room, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[0, 0, 512, 400]'

  region2_prompt='[a <dogA1> <dogA2>, sit, in a living room, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[60, 501, 350, 706]'

  region3_prompt='[a <catA1> <catA2>, sit, in a living room, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[57, 692, 343, 940]'

  region4_prompt='[a <table1> <table2>, in a living room, 4K, high quality, high resolution, best quality]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[280, 423, 508, 983]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}"

  keypose_condition='datasets/validation_spatial_condition/characters-objects/harry+catA+dogA_pose.png'
  keypose_adaptor_weight=0.8
  region_keypose_adaptor_weight=""

  sketch_condition='datasets/validation_spatial_condition/characters-objects/harry+catA+dogA_sketch.png'
  sketch_adaptor_weight=0.5
  region_sketch_adaptor_weight="${region2}-0.8|${region3}-0.8|${region4}-0.8"

  python inference/mix_of_show_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model="${combined_model_root}/${expdir}/combined_model.pth" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adpator/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --region_sketch_adaptor_weight="${region_sketch_adaptor_weight}" \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adpator/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --save_dir="results/multi-concept/${expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=14
fi

#---------------------------------------------sample people_chair_table-------------------------------------------
people_chair_table=1

if [ ${people_chair_table} -eq 1 ]
then
  keypose_condition='datasets/validation_spatial_condition/characters-objects/bengio+lecun+chair_pose.png'
  keypose_adaptor_weight=0.8

  sketch_condition='datasets/validation_spatial_condition/characters-objects/bengio+lecun+chair_sketch.png'
  sketch_adaptor_weight=0.5

  context_prompt='two men, two chairs, and a table, in a living room, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[<lecun1> <lecun2> sit on a <chair1> <chair2>, wearing a suit, in a living room, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[31, 2, 512, 376]'

  region2_prompt='[serious <bengio1> <bengio2> sit on a <chair1> <chair2>, wearing a suit, in a living room, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[smile, ${context_neg_prompt}]"
  region2='[30, 644, 506, 1011]'

  region3_prompt='[a <vase1> <vase2>, in a living room, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[110, 438, 342, 568]'

  region4_prompt='[a <table1> <table2>, in a living room, 4K, high quality, high resolution, best quality]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[290, 344, 488, 674]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}"

  python inference/mix_of_show_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
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

#---------------------------------------------sample lecun-------------------------------------------
sample_lecun=1

if [ ${sample_lecun} -eq 1 ]
then
  keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose/bengio_lecun_bengio.png'
  keypose_adaptor_weight=1.0

  sketch_condition=''
  sketch_adaptor_weight=0.5

  context_prompt='three men, standing near a lake, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[<bengio1> <bengio2>, standing near a lake, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[6, 51, 512, 293]'

  region2_prompt='[a <lecun1> <lecun2>, standing near a lake, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[1, 350, 512, 618]'

  region3_prompt='[a <hinton1> <hinton2>, standing near a lake, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[3, 657, 512, 923]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"

  python inference/mix_of_show_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
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

#---------------------------------------------sample potter-------------------------------------------
sample_potter=1

if [ ${sample_potter} -eq 1 ]
then
  keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose/harry_hermione_thanos.png'
  keypose_adaptor_weight=1.0

  sketch_condition=''
  sketch_adaptor_weight=0.5

  context_prompt='a woman, a man and a strong monster, near the castle, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <potter1> <potter2>, in hogwarts school uniform, holding hands, near the castle, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[2, 3, 512, 261]'

  region2_prompt='[a <hermione1> <hermione2>, in hogwarts school uniform, holding hands, near the castle, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[7, 207, 512, 460]'

  region3_prompt='[a <thanos1> <thanos2>, near the castle, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[1, 651, 512, 996]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"

  python inference/mix_of_show_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
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

#---------------------------------------------sample chair-------------------------------------------
sample_chair=1

if [ ${sample_chair} -eq 1 ]
then
  keypose_condition=''
  keypose_adaptor_weight=1.0

  sketch_condition='datasets/validation_spatial_condition/multi-objects/two_chair_table_vase.jpg'
  sketch_adaptor_weight=0.5

  context_prompt='two chairs, a table and a vase, in a living room, 4K, high quality, high resolution, best quality'
  context_neg_prompt='low quality, low resolution'

  region1_prompt='[a <chair1> <chair2>, in a living room, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[150, 6, 463, 293]'

  region2_prompt='[a <vase1> <vase2>, in a living room, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[53, 438, 302, 565]'

  region3_prompt='[a <chair1> <chair2>, in a living room, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[160, 724, 457, 1002]'

  region4_prompt='[a <table1> <table2>, in a living room, 4K, high quality, high resolution, best quality]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[248, 344, 468, 664]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}"

  python inference/mix_of_show_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
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
    --seed=641
fi

#---------------------------------------------sample dog-------------------------------------------
sample_dog=1

if [ ${sample_dog} -eq 1 ]
then
  keypose_condition=''
  keypose_adaptor_weight=1.0

  sketch_condition='datasets/validation_spatial_condition/multi-objects/dogA_catA_dogB.jpg'
  sketch_adaptor_weight=0.8

  context_prompt='two dogs and a cat, on the grass, under the sunset, animal photography, 4K, high quality, high resolution, best quality'
  context_neg_prompt='dark, low quality, low resolution'

  region1_prompt='[a <dogB1> <dogB2>, on the grass, under the sunset, animal photography, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[160, 76, 505, 350]'

  region2_prompt='[a <catA1> <catA2>, on the grass, under the sunset, animal photography, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[162, 370, 500, 685]'

  region3_prompt='[a <dogA1> <dogA2>, on the grass, under the sunset, animal photography, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[134, 666, 512, 1005]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"

  python inference/mix_of_show_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
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
    --seed=641
fi
