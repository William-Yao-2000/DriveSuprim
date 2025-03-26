import yaml


# define a custom representer for strings
def quoted_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')


yaml.add_representer(str, quoted_presenter)

N = 16
root = '/mnt/f/e2e/navsim2/navsim/planning/script/config/common/train_test_split/scene_filter'
tgt_yaml = 'navtest'

# Load the original YAML file
with open(f'{root}/{tgt_yaml}.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Split the tokens list into N equal parts
tokens = data['tokens']
split_tokens = [tokens[i:i + len(tokens) // N] for i in range(0, len(tokens), len(tokens) // N)]

# Generate and save sub-files
for i, token_part in enumerate(split_tokens):
    data['tokens'] = token_part
    with open(f'{root}/{tgt_yaml}_sub{i + 1}.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
