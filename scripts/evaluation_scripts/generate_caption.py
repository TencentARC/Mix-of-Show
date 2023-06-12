import argparse
import json
import os
from tqdm import tqdm

class_mapping = {
    '<catA1> <catA2>': 'cat',
    '<dogA1> <dogA2>': 'dog',
    '<dogB1> <dogB2>': 'dog',
    '<table1> <table2>': 'table',
    '<vase1> <vase2>': 'vase',
    '<chair1> <chair2>': 'chair',
    '<catA> cat': 'cat',
    '<chair> chair': 'chair',
    '<dogA> dog': 'dog',
    '<dogB> dog': 'dog',
    '<table> table': 'table',
    '<vase> vase': 'vase',
    '<catA>': 'cat',
    '<chair>': 'chair',
    '<dogA>': 'dog',
    '<dogB>': 'dog',
    '<table>': 'table',
    '<vase>': 'vase',
    '<bengio1> <bengio2>': 'man',
    '<hermione1> <hermione2>': 'woman',
    '<hinton1> <hinton2>': 'man',
    '<lecun1> <lecun2>': 'man',
    '<potter1> <potter2>': 'man',
    '<thanos1> <thanos2>': 'man',
    '<bengio> man': 'man',
    '<hermione> woman': 'woman',
    '<hinton> man': 'man',
    '<lecun> man': 'man',
    '<potter> man': 'man',
    '<thanos> man': 'man',
    '<bengio>': 'man',
    '<hermione>': 'woman',
    '<hinton>': 'man',
    '<lecun>': 'man',
    '<potter>': 'man',
    '<thanos>': 'man',
    '<pyramid1> <pyramid2>': 'pyramid',
    '<rock1> <rock2>': 'rock',
    '<pyramid> pyramid': 'pyramid',
    '<rock> rock': 'rock',
    '<pyramid>': 'pyramid',
    '<rock>': 'rock'
}


def convert_image2jsoncaption(image_dir, json_path):
    file_list = list(os.listdir(image_dir))
    json_file = {}
    for file in tqdm(file_list):
        caption = file.split('---')[0].replace('_', ' ')
        for k, v in class_mapping.items():
            caption = caption.replace(k, v)
        json_file.update({file: caption})

    with open(json_path, 'w') as file:
        json.dump(json_file, file)


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--image_dir', help='image_dir', required=True, type=str)
    parser.add_argument('--json_path', help='save json path', required=True, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert_image2jsoncaption(args.image_dir, args.json_path)
