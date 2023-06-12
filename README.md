# Mix-of-Show

Official codes for Mix-of-Show. This branch is for academic research, including paper results, evaluation, and comparison methods. For application purpose, please refer to [main branch](https://github.com/TencentARC/Mix-of-Show/tree/main) (simplified codes, memory optimization and any improvements verified in research branch).

**[Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models](https://arxiv.org/abs/2305.18292)**
<br/>
[Yuchao Gu](https://ycgu.site/), [Xintao Wang](https://xinntao.github.io/), [Jay Zhangjie Wu](https://zhangjiewu.github.io/), [Yunjun Shi](https://yujun-shi.github.io/), [Yunpeng Chen](https://cypw.github.io/), Zihan Fan, Wuyou Xiao, [Rui Zhao](https://ruizhaocv.github.io/), Shuning Chang, [Weijia Wu](https://weijiawu.github.io/), [Yixiao Ge](https://geyixiao.com/), Ying Shan, [Mike Zheng Shou](https://sites.google.com/view/showlab)
<br/>

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://showlab.github.io/Mix-of-Show/)[![arXiv](https://img.shields.io/badge/arXiv-2305.18292-b31b1b.svg)](https://arxiv.org/abs/2305.18292)



![overview](./README.assets/overview.gif)

## 📋 Results

### Single-Concept Sample Results

![single_concept](./README.assets/single_concept.jpg)

### Multi-Concept Sample Results

------

#### **Real-World Concept Results**

![real_multi_result](./README.assets/real_multi_result.jpg)

------

#### **Anime Concept Results**

![anime_multi_result](./README.assets/anime_multi_result.jpg)

## 🚩 Updates/Todo List

- [ ] Release Main Branch for Application (memory optimization, simplified codes).
- [ ] Release Colab Demo.
- [ ] Update Docs.
- [x] Jun. 12, 2023. Research Code Released.



## :wrench: Dependencies and Installation

- Python >= 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- Diffusers==0.14.0
- [PyTorch >= 1.12](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Option: Linux

### Installation

1. Install diffusers==0.14.0 (with T2I-Adapter support), credit to **[diffusers-t2i-adapter](https://github.com/HimariO/diffusers-t2i-adapter)** and **[T2I-Adapter-for-Diffusers](https://github.com/haofanwang/T2I-Adapter-for-Diffusers)**. We slightly simplify the installation steps.

    ```bash
    # Clone diffusers==0.14.0 with T2I-Adapter support
    git clone git@github.com:guyuchao/diffusers-t2i-adapter.git

    # switch to T2IAdapter-for-mixofshow
    git switch T2IAdapter-for-mixofshow

    # install from source
    pip install .
    ```


2. Clone repo & install

   ```bash
   git clone https://github.com/TencentARC/Mix-of-Show.git
   cd Mix-of-Show

   python setup.py install
   ```



## ⏬ Pretrained Model and Data Preparation

### Pretrained Model Preparation

We adopt the [ChilloutMix](https://civitai.com/models/6424/chilloutmix) for real-world concepts, and [Anything-v4](https://huggingface.co/andite/anything-v4.0) for anime concepts.

```bash
git clone https://github.com/TencentARC/Mix-of-Show.git

cd experiments/pretrained_models

# Diffusers-version ChilloutMix
git-lfs clone https://huggingface.co/windwhinny/chilloutmix.git

# Diffusers-version Anything-v4
git-lfs clone https://huggingface.co/andite/anything-v4.0.git

mkdir t2i_adapter
cd t2i_adapter

# sketch/openpose adapter of T2I-Adapter
wget https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_sketch_sd14v1.pth
wget https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_openpose_sd14v1.pth
```

### Data Preparation

Note: Data selection and tagging are important in single-concept tuning. We strongly recommend checking the data processing in [sd-scripts](https://github.com/kohya-ss/sd-scripts). **In our ED-LoRA, we do not require any regularization dataset.** For other comparison methods such as Dreambooth and Custom Diffusion, please prepare the regularization dataset according to their suggestions. Additionally, specify the regularization dataset in the [dataset/data_cfgs/dreambooth](datasets/data_cfgs/dreambooth) and [dataset/data_cfgs/custom_diffusion](datasets/data_cfgs/custom_diffusion).

The detailed dataset preparation steps can refer to [Dataset.md](docs/Dataset.md).

### Paper Resources

If you want to quickly reimplement our methods, we provide the following resources used in the paper.

<table>
<tr>
    <th>Paper Resources</th>
    <td style="text-align: center;">Concept Datasets</td>
    <td style="text-align: center;">Single-Concept Tuned ED-LoRAs</td>
    <td style="text-align: center;">Multi-Concept Fused Model</td>
		<td style="text-align: center;">Partial Sampled Results (for aligning evaluation metrics)</td>
</tr>
<tr>
    <th>Download Link</td>
    <td style="text-align: center;"><a href="https://drive.google.com/drive/folders/1y_02-TkX07HuLI_Yl64AWyTzJ-VWBR2b?usp=sharing">Google Drive</a></td>
    <td style="text-align: center;"><a href="https://drive.google.com/drive/folders/1x-SbedU1kXSsf64IlwF-jCMD04gKZ9Tq?usp=sharing">Google Drive</a></td>
    <td style="text-align: center;"><a href="https://drive.google.com/drive/folders/1QcwS3WYgq3qrpzqeKWpYqFKSOSoMm-tU?usp=sharing">Google Drive</a></td>
    <td style="text-align: center;"><a href="https://drive.google.com/drive/folders/1QiAlhZAN4-d4eTD71E9sZEXw4WJmZkNH?usp=sharing">Google Drive</a></td>
</tr>
</table>


After downloading, the path should be arranged as follows:

```
Mix-of-Show
├── mixofshow
├── scripts
├── options
├── experiments
│   ├── MixofShow_Results
│   │   ├── EDLoRA_Models
│   │   ├── Fused_Models
│   │   ├── Sampled_Results
│   ├── pretrained_models
│   │   ├── anything-v4.0
│   │   ├── chilloutmix
│   │   ├── t2i_adpator/t2iadapter_*_sd14v1.pth
├── datasets
│   ├── data
│   │   ├── characters/
│   │   ├── objects/
│   │   ├── scenes/
│   ├── data_cfgs/MixofShow
│   │   ├── single-concept # specify data path to train single-concept edlora
│   │   ├── multi-concept # specify model path to merge multiple edlora
│   ├── benchmark_prompts # benchmark prompts for calculating evaluation metrics
│   ├── validation_prompts # validation prompts during concept tuning
│   ├── ...
```



## :computer: Single-Client Concept Tuning

### Step 1: Modify the Config

Before tuning, it is essential to specify the data paths and adjust certain hyperparameters in the corresponding config file. **If you want to reimplement our results, just use the default config.** Followings are some basic config settings to be modified. For more detailed information on each config item, please refer to [Config.md](https://chat.openai.com/c/docs/Config.md).

```yaml
datasets:
  train:
    # Concept data config
    concept_list: datasets/data_cfgs/edlora/single-concept/characters/anime/hina_amano.json
    replace_mapping:
      <TOK>: <hina1> <hina2> # concept new token
  val_vis:
    # Validation prompt for visualization during tuning
    prompts: datasets/validation_prompts/single-concept/characters/test_girl.txt
    replace_mapping:
      <TOK>: <hina1> <hina2> # Concept new token

network_g:
  new_concept_token: <hina1>+<hina2> # Concept new token, use "+" to connect
  initializer_token: <rand-0.013>+girl
  # Init token, only need to revise the later one based on the semantic category of given concept

val:
  val_freq: !!float 1000 # How many iters to make a visualization during tuning
  compose_visualize: true # Compose all samples into a large grid figure for visualization
  vis_embedding: true # Visualize embedding (without LoRA weight shift)
```

### Step 2: Start Tuning

We tune each concept with 2 A100 GPU (5~10 minutes).

```bash
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=2234 mixofshow/train.py \
-opt options/train/edlora/characters/anime/train_hina.yml --launcher pytorch
```

**Note**: The process of learning embeddings is not stable even with the same device and same random seed, necessitating more attempts and hyperparameter tuning. However, once ED-LoRA is tuned, the fusion process of multiple ED-LoRAs remains stable. Therefore, more effort should be directed towards creating a high-quality ED-LoRA. We recommend enabling embedding visualization and verifying whether the embeddings encode the essence of the given concept within the pretrained model domain.

### Step 3: Sample

After tuning, specify the model path in test config, and run following command.

```bash
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=2234 mixofshow/test.py \
-opt options/test/edlora/characters/anime/test_hina.yml --launcher pytorch
```



## :computer: Center-Node Concept Fusion

### Step 1: Collect Concept Models

Collect all concept models you want to extend the pretrained model and modify the config in **datasets/data_cfgs/MixofShow/multi-concept/real/*** accordingly.

```yaml
[
    {
        "lora_path": "experiments/EDLoRA_Models/Base_Chilloutmix/characters/edlora_potter.pth", # ED-LoRA path
        "unet_alpha": 1.0, # usually use full identity = 1.0
        "text_encoder_alpha": 1.0, # usually use full identity = 1.0
        "concept_name": "<potter1> <potter2>" # new concept token
    },
    {
        "lora_path": "experiments/EDLoRA_Models/Base_Chilloutmix/characters/edlora_hermione.pth",
        "unet_alpha": 1.0,
        "text_encoder_alpha": 1.0,
        "concept_name": "<hermione1> <hermione2>"
    },

    ... # keep adding new concepts for extending the pretrained models
]
```

### Step 2: Gradient Fusion

For example, we fuse 14 concept with 1 A100 GPU (50 minutes).

```bash
export config_file="potter+hermione+thanos+hinton+lecun+bengio+catA+dogA+chair+table+dogB+vase+pyramid+rock_chilloutmix"

python scripts/mixofshow_scripts/Gradient_Fusion_EDLoRA.py \
    --concept_cfg="datasets/data_cfgs/MixofShow/multi-concept/real/${config_file}.json" \
    --save_path="experiments/composed_edlora/chilloutmix/${config_file}" \
    --pretrained_models="experiments/pretrained_models/chilloutmix" \
    --optimize_textenc_iters=500 \
    --optimize_unet_iters=50
```

### Step 3: Sample

Download our fused model on ChilloutMix (extending 14 customized concepts) and Anythingv4 (extending 5 customized concepts).

**Single-concept sampling from fused model:**

```bash
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=2234 mixofshow/test.py \
-opt options/test/MixofShow/fused_model/characters/real/fused_model_bengio.yml --launcher pytorch
```

**Regionally controllable multi-concept sampling:**

```bash
bash scripts/mixofshow_scripts/paper_result_scripts/mix_of_show_anime.sh
bash scripts/mixofshow_scripts/paper_result_scripts/mix_of_show_real.sh
```



## :straight_ruler: Evaluation

The evaluation of our method are based on two metrics: *<u>text-alignment</u>* and *<u>image-alignment</u>* following [Custom Diffusion](https://arxiv.org/abs/2212.04488).

The evaluation prompts are provided in **datasets/benchmark_prompts**. For each concept, we will generate 1000 images (20 prompts * 50 images per prompt).

Modify the path in **scripts/evaluation_scripts/evaluation.sh** and run the following command on our provided "cat" sampled results.

```bash
export image_dir="experiments/MixofShow_Results/Sampled_Results/fused_model/fused_model_catA/visualization/PromptDataset/iters_fused_model_catA"
export json_file="experiments/MixofShow_Results/Sampled_Results/fused_model/fused_model_catA.json"
export ref_image_dir="datasets/data/objects/real/cat/catA/image"

# generate caption from sampled images filename
python scripts/evaluation_scripts/generate_caption.py --image_dir ${image_dir} --json_path ${json_file}

# text-alignment, should get CLIPScore (Text-Alignment): 0.8010
python scripts/evaluation_scripts/clipscore-main/clipscore.py ${json_file} ${image_dir}

# image-alignment, should get CLIPScore (Image-Alignment): 0.8519
python scripts/evaluation_scripts/clipscore-main/clipscore_image_alignment.py ${ref_image_dir} ${image_dir}
```



## 📜 License and Acknowledgement

This project is released under the [Apache 2.0 license](LICENSE).<br>
This codebase builds on [diffusers](https://github.com/huggingface/diffusers). Thanks for open-sourcing! Besides, we acknowledge following amazing open-sourcing projects:

- LoRA for Diffusion Models (https://github.com/cloneofsimo/lora, https://github.com/kohya-ss/sd-scripts).


- Custom Diffusion (https://github.com/adobe-research/custom-diffusion).


- T2I-Adapter (https://github.com/TencentARC/T2I-Adapter).



## 🌏 Citation

```bibtex
@article{gu2023mixofshow,
    title={Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models},
    author={Gu, Yuchao and Wang, Xintao and Wu, Jay Zhangjie and Shi, Yujun and Chen Yunpeng and Fan, Zihan and Xiao, Wuyou and Zhao, Rui and Chang, Shuning and Wu, Weijia and Ge, Yixiao and Shan Ying and Shou, Mike Zheng},
    journal={arXiv preprint arXiv:2305.18292},
    year={2023}
}
```



## 📧 Contact

If you have any questions and improvement suggestions, please email Yuchao Gu (yuchaogu9710@gmail.com), or open an issue.
