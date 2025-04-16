import os
import pickle

traj_root = os.getenv('NAVSIM_TRAJPDM_ROOT')

if __name__ == '__main__':
    import pdb; pdb.set_trace()
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # navtrain subset文件夹的前缀名
    old_out_dir = 'ori/'

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
