import os
import pickle
from collections import defaultdict
import shelve
from tqdm import tqdm

traj_root = os.getenv('NAVSIM_TRAJPDM_ROOT')

if __name__ == '__main__':
    import pdb; pdb.set_trace()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rot', type=int, default=30, help='rotation augmentation')
    parser.add_argument('--trans', type=int, default=0, help='translation augmentation')
    parser.add_argument('--va', type=float, default=0.0, help='view angle augmentation')
    parser.add_argument('--percentage', type=float, default=0.5, help='percentage of data to use')
    args = parser.parse_args()
    
    rot, trans, va = args.rot, args.trans, args.va
    percentage = args.percentage
    all_data = defaultdict(list)

    old_out_dir_prefix = 'random_aug/'
    parts = []
    parts.append(f'rot_{rot}-trans_{trans}')
    if va == 0:
        va = int(va)
    parts.append(f'va_{va}')
    
    if len(parts) > 0:
        old_out_dir_prefix += '-'.join(parts)
    old_out_dir_prefix += f'-p_{percentage}'

    out_dir = old_out_dir_prefix + '-ensemble'
    pkl_file = f'{traj_root}/{out_dir}/navtrain_ensemble.pkl'
    pkl_data = pickle.load(open(pkl_file, 'rb'))
    os.makedirs(f'{traj_root}/{out_dir}/split_pickles', exist_ok=True)
    for k, v in tqdm(pkl_data.items()):
        with open(f'{traj_root}/{out_dir}/split_pickles/{k}.pkl', 'wb') as f:
            pickle.dump(v, f)
