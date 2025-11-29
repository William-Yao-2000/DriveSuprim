import json, yaml
import argparse
import random
from tqdm import tqdm
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rot', default=0, type=int, help='the rotation angle boundary (degree)')
    parser.add_argument('--percentage', default=0.5, type=float, help='the percentage of augmented data')
    parser.add_argument('--begin_seed', default=2024, type=int, help='random seed')
    parser.add_argument('--end_seed', default=2026, type=int, help='random seed')
    args = parser.parse_args()

    with open('navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    tokens = data['tokens']
    ensemble_data = dict()
    ensemble_data['param'] = {'rot': args.rot, 'per': args.percentage}
    ensemble_data['tokens'] = defaultdict(list)

    old_out_dir = os.path.join(os.getenv('NAVSIM_EXP_ROOT'), 'offline_files/training_ego_aug_files')
    filename_parts = []
    filename_parts.append(f'rot_{args.rot}')
    filename_parts.append(f'p_{args.percentage}')
    filename_prefix = '-'.join(filename_parts)
    for seed in range(args.begin_seed, args.end_seed+1):
        
        filename = filename_prefix + f'-seed_{seed}.json'

        print(filename)
        with open(os.path.join(old_out_dir, filename), 'r') as f:
            data = json.load(f)
        assert data['param']['rot'] == args.rot and data['param']['per'] == args.percentage, \
            f"not aligned file! This augmentation file rot: {data['param']['rot']}, per: {data['param']['per']}"
        for k in tokens:
            ensemble_data['tokens'][k].append(data['tokens'][k])
    
    with open(f"{old_out_dir}/{filename_prefix}-ensemble.json", 'w') as f:
        json.dump(ensemble_data, f)


if __name__ == '__main__':
    main()