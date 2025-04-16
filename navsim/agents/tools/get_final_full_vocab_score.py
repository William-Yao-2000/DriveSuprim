import os
import pickle

traj_root = os.getenv('NAVSIM_TRAJPDM_ROOT')

if __name__ == '__main__':
    import pdb; pdb.set_trace()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rot', type=int, default=30, help='rotation augmentation')
    parser.add_argument('--trans', type=int, default=0, help='translation augmentation')
    parser.add_argument('--va', type=float, default=0, help='view angle augmentation')
    parser.add_argument('--percentage', type=float, default=0.5, help='percentage of data to use')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    
    rot, trans, va = args.rot, args.trans, args.va
    percentage = args.percentage
    seed = args.seed
    # navtrain subset文件夹的前缀名
    old_out_dir = 'random_aug/'
    parts = []
    parts.append(f'rot_{rot}-trans_{trans}')
    va = int(va)
    parts.append(f'va_{va}')
    
    if len(parts) > 0:
        old_out_dir += '-'.join(parts)
    old_out_dir += f'-p_{percentage}'
    old_out_dir += f'-seed_{seed}'

    # 最后新文件夹的名字，存入整个navtrain集合
    new_out_dir = f'{old_out_dir}/vocab_score_8192_navtrain_final'
    os.makedirs(f'{traj_root}/{new_out_dir}', exist_ok=True)

    ins = [f'ngc_sub{i}' for i in range(1,13)]
    out = 'navtrain.pkl'

    result = {}
    for in_pkl in ins:
        curr_pickle_path = f'{traj_root}/{old_out_dir}/vocab_score_8192_navtrain_{in_pkl}/navtrain_{in_pkl}.pkl'
        curr_pickle = pickle.load(open(curr_pickle_path, 'rb'))
        print("------------")
        print(curr_pickle_path, len(curr_pickle))
        for k, v in curr_pickle.items():
            result[k] = v
    print(f'Length: {len(result)}')
    pickle.dump(result, open(f'{traj_root}/{new_out_dir}/{out}', 'wb'))
