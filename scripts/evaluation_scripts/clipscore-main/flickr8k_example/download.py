'''Downloads and preprocesses the Flickr8K corpus.

For more details about particular choices made when computing
correlations, please see appendix A in:

CLIPScore:
A Reference-free Evaluation Metric for Image Captioning
by Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, Yejin Choi
https://arxiv.org/abs/2104.08718
'''
import collections
import json
import numpy as np
import os
from google_drive_downloader import GoogleDriveDownloader as gdd


def process_flickr8k():
    if not os.path.exists('flickr8k'):
        os.makedirs('flickr8k')

    if not os.path.exists('flickr8k/Flickr8k_Dataset.zip'):
        gdd.download_file_from_google_drive(
            file_id='1WNY8pV-u8xtBYBVal03qwjQs4VKurUZn',
            dest_path='./flickr8k/Flickr8k_Dataset.zip',
            unzip=True,
            showsize=True)
    if not os.path.exists('flickr8k/Flickr8k_text.zip'):
        gdd.download_file_from_google_drive(
            file_id='1ljB7DR-YM-q9WKnHDW5dHjauK029B2s6',
            dest_path='./flickr8k/Flickr8k_text.zip',
            unzip=True,
            showsize=True)

    # flickr8k #
    flickr8k_image2ann = collections.defaultdict(list)
    captionid2caption = {}

    with open('flickr8k/Flickr8k.token.txt') as f:
        for line in f:
            image, ann = line.strip().split('\t')
            flickr8k_image2ann[image.split('#')[0]].append(ann)
            captionid2caption[image] = ann

    flickr8k_image2ann = {k.split('.')[0]: v for k, v in flickr8k_image2ann.items()}

    all_8k_expert, all_8k_crowdflower = [], []

    total_ann = 0
    with open('flickr8k/CrowdFlowerAnnotations.txt') as f:
        for line in f:
            image_id, caption_id, yes_prec, n_yes, n_no = line.strip().split('\t')
            all_8k_crowdflower.append({
                'image_id': image_id,
                'caption_id': caption_id,
                'caption': captionid2caption[caption_id],
                'n_yes': int(n_yes),
                'n_no': int(n_no),
                'yes_prec': float(yes_prec),
                'image_filepath': 'Flicker8k_Dataset/{}'.format(image_id)
            })
            assert np.abs(float(yes_prec) - int(n_yes) / (int(n_yes) + int(n_no))) < 10E-6
            total_ann += (int(n_yes) + int(n_no))

    all_index = {}
    for d in all_8k_crowdflower:
        if d['image_id'] not in all_index:
            all_index[d['image_id']] = {
                'human_judgement': [],
                'image_id': d['image_id'],
                'image_path': d['image_filepath'],
                'ground_truth': [x for x in flickr8k_image2ann[d['image_id'].split('.')[0]]]
            }

        if d['caption'] in all_index[d['image_id']]['ground_truth']:
            all_index[d['image_id']]['ground_truth'].remove(d['caption'])

        all_index[d['image_id']]['human_judgement'].append({
            'image_id': d['image_id'],
            'image_path': d['image_filepath'],
            'caption': d['caption'],
            'rating': d['yes_prec']
        })

    print('For crowdflower, we are dumping {} judgments between {} images'.format(
        len(all_8k_crowdflower), len(all_index)))

    with open('flickr8k/crowdflower_flickr8k.json', 'w') as f:
        f.write(json.dumps(all_index))

    skip = 0
    with open('flickr8k/ExpertAnnotations.txt') as f:
        for line in f:
            image_id, caption_id, ex1, ex2, ex3 = line.strip().split('\t')
            caption = captionid2caption[caption_id]
            # we will skip the ones in the refs following the SPICE paper.
            if caption in flickr8k_image2ann[image_id.split('.')[0]]:
                skip += 1
                continue
            all_8k_expert.append({
                'image_id': image_id.split('.')[0],
                'image_filepath': 'Flicker8k_Dataset/{}'.format(image_id),
                'caption': caption,
                'expert1': float(ex1),
                'expert2': float(ex2),
                'expert3': float(ex3)
            })

    all_index = {}
    for d in all_8k_expert:
        if d['image_id'] not in all_index:
            all_index[d['image_id']] = {
                'human_judgement': [],
                'image_id': d['image_id'],
                'image_path': d['image_filepath'],
                'ground_truth': flickr8k_image2ann[d['image_id']]
            }

        all_index[d['image_id']]['human_judgement'].append({
            'image_id': d['image_id'],
            'image_path': d['image_filepath'],
            'caption': d['caption'],
            'rating': d['expert1']
        })
        all_index[d['image_id']]['human_judgement'].append({
            'image_id': d['image_id'],
            'image_path': d['image_filepath'],
            'caption': d['caption'],
            'rating': d['expert2']
        })
        all_index[d['image_id']]['human_judgement'].append({
            'image_id': d['image_id'],
            'image_path': d['image_filepath'],
            'caption': d['caption'],
            'rating': d['expert3']
        })

    print('For expert, we are dumping {} judgments between {} images'.format(len(all_8k_expert) * 3, len(all_index)))

    with open('flickr8k/flickr8k.json', 'w') as f:
        f.write(json.dumps(all_index))


def main():
    if not os.path.exists('flickr8k/flickr8k.json'):
        process_flickr8k()


if __name__ == '__main__':
    main()
