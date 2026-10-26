import pandas as pd

# 读取Excel文件
df = pd.read_excel('历史策略独有数据_20251223.xlsx')  # 请将'your_file.csv'替换为您的实际文件路径

# 定义需要读取的列
target_columns = ['file2_P1_Flow', 'file2_P2_Flow', 'file2_P3_Flow', 'file2_P4_Flow', 'file2_P5_Flow']

# 定义各列的上限值，顺序与target_columns中的列顺序对应
upper_limits = [1460, 1460, 1460, 1460, 2910]

# 方法一：直接使用条件判断
# 检查每一列是否都小于等于对应的上限值
condition = True
for col, limit in zip(target_columns, upper_limits):
    condition = condition & (df[col] <= limit)

# 统计同时满足所有条件的数据行个数
valid_row_count = condition.sum()
total_rows = len(df)

print(f"五列数据都满足不超过上限值的数据行个数为: {valid_row_count}")

# 1. 统计至少有一列不满足上限的数据行占比
# 创建条件判断每一列是否超过上限
any_condition = False
for col, limit in zip(target_columns, upper_limits):
    any_condition = any_condition | (df[col] > limit)

# 计算至少有一列不满足条件的行数
any_invalid_count = any_condition.sum()
any_invalid_ratio = any_invalid_count / total_rows if total_rows > 0 else 0

print(f"\n至少有一列数据不满足上限值的数据行个数为: {any_invalid_count}")
print(f"至少有一列数据不满足上限值的数据行占比为: {any_invalid_ratio:.2%}")

# 2. 统计所有数据列都满足上限但Flow_Deviation大于25的数据行占比
# 先找出所有列都满足条件的数据行
all_valid_condition = True
for col, limit in zip(target_columns, upper_limits):
    all_valid_condition = all_valid_condition & (df[col] <= limit)

# 在这些行中找出Flow_Deviation大于25的行
deviation_condition = (all_valid_condition) & (df['file2_Flow_Deviation'] > 25)
deviation_invalid_count = deviation_condition.sum()
deviation_invalid_ratio = deviation_invalid_count / total_rows if total_rows > 0 else 0

print(f"\n所有数据列满足上限但Flow_Deviation>25的数据行个数为: {deviation_invalid_count}")
print(f"所有数据列满足上限但Flow_Deviation>25的数据行占比为: {deviation_invalid_ratio:.2%}")

# 方法二：使用all()和any()函数进行验证
# 创建布尔DataFrame判断每个元素是否满足条件
bool_df = pd.DataFrame()
for col, limit in zip(target_columns, upper_limits):
    bool_df[col] = df[col] <= limit

# 使用all(axis=1)判断每一行是否所有列都满足条件
all_conditions_met = bool_df.all(axis=1)
valid_row_count_alt = all_conditions_met.sum()

# 使用any(axis=1)判断每一行是否至少有一列不满足条件
any_condition_invalid = (~bool_df).any(axis=1)
any_invalid_count_alt = any_condition_invalid.sum()

print(f"\n使用方法二验证 - 所有列满足上限的行数: {valid_row_count_alt}")
print(f"使用方法二验证 - 至少一列不满足上限的行数: {any_invalid_count_alt}")

# 显示总体统计信息
print(f"\n=== 总体统计 ===")
print(f"数据总行数: {total_rows}")
print(f"所有列都满足上限的行数: {valid_row_count} ({valid_row_count/total_rows:.2%})")
print(f"至少一列不满足上限的行数: {any_invalid_count} ({any_invalid_ratio:.2%})")
print(f"所有列满足上限但Deviation>25的行数: {deviation_invalid_count} ({deviation_invalid_ratio:.2%})")

# 可选：查看满足条件的部分数据
if valid_row_count > 0:
    valid_data = df[target_columns + ['file2_Flow_Deviation']][all_valid_condition]
    print(f"\n所有列满足上限的数据预览(前5行):")
    print(valid_data.head())