import json, yaml
import argparse
import random
from tqdm import tqdm
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rot', default=0, type=int, help='the rotation angle boundary (degree)')
    parser.add_argument('--trans', default=0, type=int)
    parser.add_argument('--va', default=0, type=float, help='the boundary of disturbance proportion in velocity and acceleration')
    parser.add_argument('--percentage', default=0.5, type=float, help='the percentage of augmented data')
    args = parser.parse_args()

    with open('navsim/planning/script/config/common/scene_filter/navtrain.yaml', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    import pdb; pdb.set_trace()
    tokens = data['tokens']
    ensemble_data = dict()
    ensemble_data['param'] = {'rot': args.rot, 'trans': args.trans, 'va': args.va, 'per': args.percentage}
    ensemble_data['tokens'] = defaultdict(list)

    old_out_dir = os.path.join(os.getenv('NAVSIM_EXP_ROOT'), 'offline_files/training_ego_aug_files')
    filename_parts = []
    if args.rot != 0 or args.trans != 0:
        filename_parts.append(f'rot_{args.rot}-trans_{args.trans}')
    if args.va != 0:
        filename_parts.append(f'va_{args.va}')
    filename_parts.append(f'p_{args.percentage}')
    filename_prefix = '-'.join(filename_parts)
    for seed in range(2024, 2027):
        
        filename = filename_prefix + f'-seed_{seed}.json'

        print(filename)
        with open(os.path.join(old_out_dir, filename), 'r') as f:
            data = json.load(f)
        for k in tokens:
            ensemble_data['tokens'][k].append(data['tokens'][k])
    
    with open(f"{old_out_dir}/{filename_prefix}-ensemble.json", 'w') as f:
        json.dump(ensemble_data, f)


if __name__ == '__main__':
    main()