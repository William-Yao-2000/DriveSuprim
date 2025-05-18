import pandas as pd

# 文件路径
file1 = "/DATA3/yaowenhao/proj/auto_drive/navsim_workspace/exp_v2/training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/labs/test-09ep-student/2025.05.14.09.44.54.csv"
file2 = "/DATA3/yaowenhao/proj/auto_drive/navsim_workspace/exp_v2/training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/stage_layers_3-topks_256/test-05ep/2025.05.14.06.34.42.csv"

# 读取 CSV 文件（跳过第一列空索引）
df1 = pd.read_csv(file1, index_col=0)
df2 = pd.read_csv(file2, index_col=0)

# 合并两个 DataFrame，按 token 对齐
merged = pd.merge(
    df1[['token', 'score']].rename(columns={'score': 'score_file1'}),
    df2[['token', 'score']].rename(columns={'score': 'score_file2'}),
    on='token'
)

# 过滤条件：file2 的 score 比 file1 大 0.4 以上
filtered = merged[merged['score_file2'] > merged['score_file1'] + 0.4]

# 保存到新 CSV 文件
output_path = "temp-test-code/score_diff_tokens.csv"
filtered.to_csv(output_path, index=False)

print(f"共找到 {len(filtered)} 个 token 满足条件，已保存到：{output_path}")
