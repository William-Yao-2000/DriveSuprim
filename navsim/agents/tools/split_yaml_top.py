import yaml

N = 16
root = '/mnt/f/e2e/navsim2/navsim/planning/script/config/common/train_test_split'
tgt_yaml = 'navtrain'

# Load the original YAML file
with open(f'{root}/{tgt_yaml}.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Generate and save sub-files
for i in range(N):
    data['defaults'][0]['scene_filter'] = f'{tgt_yaml}_sub{i + 1}'
    with open(f'{root}/{tgt_yaml}_sub{i + 1}.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
