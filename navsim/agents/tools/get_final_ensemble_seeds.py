import os
import pickle
from collections import defaultdict

traj_root = os.getenv('NAVSIM_TRAJPDM_ROOT')

if __name__ == '__main__':
    import pdb; pdb.set_trace()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rot', type=int, default=30, help='rotation augmentation')
    parser.add_argument('--trans', type=int, default=0, help='translation augmentation')
    parser.add_argument('--va', type=float, default=0, help='view angle augmentation')
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

    for seed in range(2024, 2027):
        
        old_out_dir = old_out_dir_prefix + f'-seed_{seed}'

        old_out_dir = f'{old_out_dir}/vocab_score_8192_navtrain_final'

        curr_pickle_path = f'{traj_root}/{old_out_dir}/navtrain.pkl'
        curr_pickle = pickle.load(open(curr_pickle_path, 'rb'))
        print("------------")
        print(curr_pickle_path, len(curr_pickle))
            
        for k, v in curr_pickle.items():
            all_data[k].append(v)

    print(f'Length: {len(all_data)}')
    assert all([len(v) == 3 for v in all_data.values()])

    new_out_dir = old_out_dir_prefix + '-ensemble'
    os.makedirs(f'{traj_root}/{new_out_dir}', exist_ok=True)
    pickle.dump(all_data, open(f'{traj_root}/{new_out_dir}/navtrain_ensemble.pkl', 'wb'))
