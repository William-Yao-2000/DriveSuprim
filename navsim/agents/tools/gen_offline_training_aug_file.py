import json, yaml
import argparse
import random
from tqdm import tqdm
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rot', default=0, type=int, help='the rotation angle boundary (degree)')
    parser.add_argument('--trans', default=0, type=int)
    parser.add_argument('--va', default=0, type=float, help='the boundary of disturbance proportion in velocity and acceleration')
    parser.add_argument('--percentage', default=0.5, type=float, help='the percentage of augmented data')
    parser.add_argument('--seed', default=2024, type=int, help='random seed')
    args = parser.parse_args()

    with open('navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    import pdb; pdb.set_trace()
    random.seed(args.seed)
    tokens = data['tokens']
    num_augmented = int(len(tokens)*args.percentage)
    aug_tokens = random.sample(tokens, num_augmented)
    assert len(aug_tokens) == num_augmented and len(set(aug_tokens)) == num_augmented
    new_data = {
        'param': {'rot': args.rot, 'trans': args.trans, 'va': args.va, 'per': args.percentage, 'seed': args.seed},
        'tokens': {},
    }
    for token in tqdm(aug_tokens):
        rot = random.uniform(-args.rot, args.rot)
        trans = random.uniform(-args.trans, args.trans)
        vel = random.uniform(-args.va, args.va)
        acc = random.uniform(-args.va, args.va)
        new_data['tokens'][token] = {'rot': rot, 'trans': trans, 'vel': vel, 'acc': acc}
    for token in tqdm(list(set(tokens)-set(aug_tokens))):
        new_data['tokens'][token] = {'rot': 0, 'trans': 0, 'vel': 0, 'acc': 0}
    
    os.makedirs(os.path.join(os.getenv('NAVSIM_EXP_ROOT'), 'offline_files/training_ego_aug_files'), exist_ok=True)
    filename_parts = []
    filename_parts.append(f'rot_{args.rot}-trans_{args.trans}')
    filename_parts.append(f'va_{int(args.va)}')
    filename_parts.append(f'p_{args.percentage}')
    filename_parts.append(f'seed_{args.seed}')
    filename = '-'.join(filename_parts) + '.json'

    with open(os.path.join(os.getenv('NAVSIM_EXP_ROOT'), f'offline_files/training_ego_aug_files/{filename}'), 'w') as f:
        json.dump(new_data, f)



if __name__ == '__main__':
    main()