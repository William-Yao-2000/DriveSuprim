import os
import pickle
from collections import defaultdict

traj_root = os.getenv('NAVSIM_TRAJPDM_ROOT')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rot', type=int, default=30, help='rotation augmentation')
    parser.add_argument('--percentage', type=float, default=0.5, help='percentage of data to use')
    parser.add_argument('--begin_seed', type=int, default=2024, help='begin seed')
    parser.add_argument('--end_seed', type=int, default=2026, help='end seed')
    parser.add_argument('--debug_split', action='store_true', help='use debug split')
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

    for seed in range(args.begin_seed, args.end_seed + 1):
        
        old_out_dir = old_out_dir_prefix + f'-seed_{seed}'
        old_out_dir = f'{old_out_dir}/vocab_score_8192_navtrain'

        if args.debug_split:
            old_out_dir += '_debug'

        curr_pickle_path = f'{traj_root}/{old_out_dir}/navtrain.pkl'

        if args.debug_split:
            curr_pickle_path = curr_pickle_path.replace('.pkl', '_debug.pkl')

        curr_pickle = pickle.load(open(curr_pickle_path, 'rb'))
        print("------------")
        print(curr_pickle_path, len(curr_pickle))
            
        for k, v in curr_pickle.items():
            all_data[k].append(v)

    print(f'Length: {len(all_data)}')
    assert all([len(v) == 3 for v in all_data.values()])

    new_out_dir = old_out_dir_prefix + '-ensemble'
    if args.debug_split:
        new_out_dir += '_debug'

    os.makedirs(f'{traj_root}/{new_out_dir}', exist_ok=True)
    if args.debug_split:
        pickle.dump(all_data, open(f'{traj_root}/{new_out_dir}/navtrain_debug_ensemble.pkl', 'wb'))
    else:
        pickle.dump(all_data, open(f'{traj_root}/{new_out_dir}/navtrain_ensemble.pkl', 'wb'))
