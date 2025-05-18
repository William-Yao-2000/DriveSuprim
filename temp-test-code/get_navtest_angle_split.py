import yaml
import pandas as pd

# 路径设置
original_yaml_path = "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
import json
token_data = json.load(open('temp-test-code/token_angle_categories.json', 'r'))

for i in range(0, 3):
    output_yaml_path = f"/DATA3/yaowenhao/proj/auto_drive/navsim_workspace/navsim_ssl_v2/navsim/planning/script/config/common/train_test_split/scene_filter/navtest_angle_split_{i}.yaml"

    import pdb; pdb.set_trace()


    if i == 1:
        token_list = token_data['2']
    elif i == 0:
        token_list = token_data['0']+token_data['1']
    else:
        token_list = token_data['3']+token_data['4']

    # 加载原始 YAML 文件
    with open(original_yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    # 替换 tokens 字段
    yaml_data['tokens'] = token_list

    # 保存为新的 YAML 文件
    with open(output_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"已将 tokens 字段替换为新的 token 列表，并保存到：{output_yaml_path}")
