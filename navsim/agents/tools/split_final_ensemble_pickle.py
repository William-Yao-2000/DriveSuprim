import os
import pickle
from collections import defaultdict
import shelve
from tqdm import tqdm

traj_root = os.getenv('NAVSIM_TRAJPDM_ROOT')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rot', type=int, default=30, help='rotation augmentation')
    parser.add_argument('--percentage', type=float, default=0.5, help='percentage of data to use')
    parser.add_argument('--debug_split', action='store_true', help='debug split')
    args = parser.parse_args()
    
    rot = args.rot
    percentage = args.percentage
    all_data = defaultdict(list)

    old_out_dir_prefix = 'random_aug/'
    parts = []
    parts.append(f'rot_{rot}')
    
    if len(parts) > 0:
        old_out_dir_prefix += '-'.join(parts)
    old_out_dir_prefix += f'-p_{percentage}'

    out_dir = old_out_dir_prefix + '-ensemble'
    if args.debug_split:
        out_dir += '_debug'
    pkl_file = f'{traj_root}/{out_dir}/navtrain_ensemble.pkl'
    if args.debug_split:
        pkl_file = f'{traj_root}/{out_dir}/navtrain_debug_ensemble.pkl'
    pkl_data = pickle.load(open(pkl_file, 'rb'))
    os.makedirs(f'{traj_root}/{out_dir}/split_pickles', exist_ok=True)
    for k, v in tqdm(pkl_data.items()):
        with open(f'{traj_root}/{out_dir}/split_pickles/{k}.pkl', 'wb') as f:
            pickle.dump(v, f)
