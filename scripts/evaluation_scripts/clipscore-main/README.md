# What's in here?

This repo contains the code for our EMNLP 2021 paper: [CLIPScore: A
Reference-free Evaluation Metric for Image
Captioning](https://arxiv.org/abs/2104.08718). CLIPScore is a metric
that you can use to evaluate the quality of an automatic image
captioning system.  In our paper, we show that CLIPScore achieves high
correlation with human judgment on literal image captioning
tasks. However, unlike BLEU or CIDEr, CLIPScore doesn't require
reference captions.

If you find the paper or this code useful, please consider citing:

```
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
```

# How do I run the code?

## Command Line

Example usage
```
> python clipscore.py example/good_captions.json example/images/
...
CLIPScore: 0.8584
```

If you include optionally some references, you will see RefCLIPScore, alongside a usual set of
caption generation evaluation metrics. The references are optional.

```
> python clipscore.py example/good_captions.json example/images/ --references_json example/refs.json
...
BLEU-1: 0.6667
BLEU-2: 0.4899
BLEU-3: 0.3469
BLEU-4: 0.0000
METEOR: 0.3444
ROUGE: 0.4280
CIDER: 0.5637
SPICE: 0.4000
CLIPScore: 0.8584
RefCLIPScore: 0.8450
```

Worse captions should get lower scores:
```
> python clipscore.py example/bad_captions.json example/images/ --references_json example/refs.json
...
BLEU-1: 0.4815
BLEU-2: 0.2404
BLEU-3: 0.1359
BLEU-4: 0.0000
METEOR: 0.1861
ROUGE: 0.3121
CIDER: 0.2790
SPICE: 0.1500
CLIPScore: 0.7153
RefCLIPScore: 0.7253
```

You can treat/report CLIPScore and RefCLIPScore similarly to the other
evaluation metrics. See the paper for more details about CLIPScore and
RefCLIPScore. Full usage options can be given by `python clipscore.py
-h`.  An example set of inputs, including a candidate json, image
directory, and references json is given this repo under `example/`

The input files are formatted as follows:

The candidates json should be a dictionary that maps from
`{"string_image_identifier": "candidate"}`, e.g.,

```
{'image1': 'an orange cat and a grey cat are lying together.',
 'image2': 'a black dog looks at the camera.'
 ...}
```

The image directory should be a directory containing the images that
act as the keys in the candidates json, e.g.,

```
images/
├── image1.jpg
└── image2.jpg
```

and, finally, the references json should be a dictionary that maps from
`{"string_image_identifier": ["list", "of", "references"]}`, e.g.,

```
{"image1": ["two cats are sleeping next to each other.",
            "a grey cat is cuddling with an orange cat on a blanket.",
	    "the orange cat is happy that the black cat is close to it."],
 "image2": ["a dog is wearing ear muffs as it lies on a carpet.",
            "a black dog and an orange cat are looking at the photographer.",
	    "headphones are placed on a dogs ears."]}
```

## MSCOCO dataset in pycocoevalcap

If you're running on the MSCOCO dataset and using the standard
evaluation toolkit, you can use our version of
[pycocoevalcap](https://github.com/jmhessel/pycocoevalcap) to evaluate.
You won't even need to download the original MSCOCO images, thanks to
a bit of magic :-)

To use `pycocoevalcap` on the MSCOCO dataset in the MSCOCO format, you
can simply use:

```
pip install git+https://github.com/jmhessel/pycocoevalcap.git
```

there is an example evaluation in that repo under
`examples/eval.py`. After pip installing, if you clone the
`pycocoeval` repo and run

```
python eval.py
```

after a bit of time, the output should be:
```
Bleu_1: 0.579
Bleu_2: 0.404
Bleu_3: 0.279
Bleu_4: 0.191
METEOR: 0.195
ROUGE_L: 0.396
CIDEr: 0.600
SPICE: 0.133
CLIPScore: 0.528
RefCLIPScore: 0.605
```

## Reproducibility notes:

- CLIPScore can run either on CPU or GPU. But, there are slight
  differences due to floating point precision. As discussed
  [here](https://github.com/openai/CLIP/issues/30#issuecomment-771099118),
  on CPU, all operations run in `float32`, but on GPU, some operations
  run in `float16`. The differences are generally small (e.g., for the
  example run above, with `example/good_captions.json` captions and
  `example/images/` images, on CPU, the output is `CLIPScore: 0.8585`,
  but on GPU, the output is `CLIPScore: 0.8584`.) *All experiments in the
  paper were run on GPU, and this code will raise a warning if you're not
  using a GPU.*

- Because CLIPScore depends on the images to compute, resizing,
  compressing, etc. can all cause slight differences in computing the
  CLIPScore. Even saving a jpg twice can result in different
  compression, because that format is lossy! To this end, we release
  the checksums of the images we used for the paper. see `checksums/`
  for more info. For the pycocoevalcap repo, we have also included the
  checksums for MSCOCO --- see
  [here](https://github.com/jmhessel/pycocoevalcap/tree/master/clipscore)
  for more info.

- The prompt we used for the text side of CLIP, as mentioned in the
  paper is ``A photo depicts" This is hard-coded into this repo. Other
  prompts will result in slightly different results, and we don't
  recommend them for the sake of reproducibility.

  ## Acknowledgment

  The authors would like to thank Jungo Kasai for being the first to use
  this repo. Thanks to Jungo, we fixed a few issues, and added some
  information about reproducibility that was missing before.
