import pandas as pd
import numpy as np

# 定义常量
# 整体效率的筛选范围
MIN_OVERALL_EFFICIENCY = 0.4
MAX_OVERALL_EFFICIENCY = 0.9

# 单个泵效率的筛选范围（运行中的泵）
MIN_PUMP_EFFICIENCY = -1.0    # 运行泵的最小效率
MAX_PUMP_EFFICIENCY = 1.0    # 运行泵的最大效率

# 千吨水电耗的筛选范围
MIN_PCTW = 100
MAX_PCTW = 200

# 定义泵开启的阈值（频率大于此值认为泵开启）
PUMP_ON_THRESHOLD = 0.0  # 频率大于0.0Hz认为泵开启

# 读取数据
df = pd.read_csv('processed_pump_data.csv')

# 1. 定义分组依据
group_columns = ['泵组', '总管瞬时流量_四舍五入', '总管压力_四舍五入', '平均吸水井液位_四舍五入']

# 2. 在每组内，找到"整体效率"最高的行
idx = df.groupby(group_columns)['整体效率'].idxmax()

# 3. 根据这些索引，从原始数据中提取出最优解的行
df_optimal = df.loc[idx].copy()

# 4. 计算Total_Flow（六泵总出口流量）
# 检查单泵流量列是否存在
pump_flow_columns = ['1号泵出口流量', '2号泵出口流量', '3号泵出口流量', 
                     '4号泵出口流量', '5号泵出口流量', '6号泵出口流量']

# 如果有泵流量数据，计算总和，并确保各泵流量不为负值
if all(col in df_optimal.columns for col in pump_flow_columns):
    # 确保每个泵的出口流量不为负值，将负值设为0
    for col in pump_flow_columns:
        df_optimal[col] = df_optimal[col].clip(lower=0)
    
    # 计算总流量
    df_optimal['Total_Flow'] = df_optimal[pump_flow_columns].sum(axis=1)
else:
    print("警告：部分泵流量列不存在，Total_Flow将设置为NaN")
    df_optimal['Total_Flow'] = 0

# 5. 计算Flow_Deviation（总管瞬时流量_原始与六泵总出口流量的差值绝对值）
if '总管瞬时流量_原始' in df_optimal.columns and 'Total_Flow' in df_optimal.columns:
    df_optimal['Flow_Deviation'] = abs(df_optimal['总管瞬时流量_原始'] - df_optimal['Total_Flow'])
else:
    print("警告：无法计算Flow_Deviation，将设置为NaN")
    df_optimal['Flow_Deviation'] = 0

# 6. 添加筛选条件
print("筛选前记录数:", len(df_optimal))

# 6.1 筛选整体效率在指定范围内
if '整体效率' in df_optimal.columns:
    df_optimal = df_optimal[(df_optimal['整体效率'] >= MIN_OVERALL_EFFICIENCY) & 
                            (df_optimal['整体效率'] <= MAX_OVERALL_EFFICIENCY)]
    print(f"整体效率在{MIN_OVERALL_EFFICIENCY}-{MAX_OVERALL_EFFICIENCY}之间筛选后记录数:", len(df_optimal))
else:
    print("警告：整体效率列不存在，跳过此筛选")

# 6.2 筛选千吨水电耗在100-200之间
if '千吨水电耗' in df_optimal.columns:
    df_optimal = df_optimal[(df_optimal['千吨水电耗'] >= MIN_PCTW) & 
                            (df_optimal['千吨水电耗'] <= MAX_PCTW)]
    print(f"千吨水电耗在{MIN_PCTW}-{MAX_PCTW}之间筛选后记录数:", len(df_optimal))
else:
    print("警告：千吨水电耗列不存在，跳过此筛选")

# 6.3 筛选每个泵的效率 - 只对开启的泵进行检查
pump_freq_columns = ['1号泵频率给定', '2号泵频率给定', '3号泵频率给定', 
                    '4号泵频率给定', '5号泵频率给定', '6号泵频率给定']

pump_efficiency_columns = ['1号泵效率', '2号泵效率', '3号泵效率', 
                          '4号泵效率', '5号泵效率', '6号泵效率']

# 检查并筛选每个泵的效率
pump_removed_counts = {}  # 记录每个泵移除的记录数
for i in range(1, 7):
    freq_col = f'{i}号泵频率给定'
    eff_col = f'{i}号泵效率'
    
    if freq_col in df_optimal.columns and eff_col in df_optimal.columns:
        # 记录筛选前的记录数
        count_before = len(df_optimal)
        
        # 判断泵是否开启：频率大于阈值
        # 注意：需要考虑NaN值
        pump_on_condition = df_optimal[freq_col].fillna(0) > PUMP_ON_THRESHOLD
        
        # 创建筛选条件：
        # 1. 如果泵未开启（频率<=阈值），则通过
        # 2. 如果泵开启但效率为NaN，则通过
        # 3. 如果泵开启且效率有值，则必须在有效范围内
        condition = (~pump_on_condition) | (df_optimal[eff_col].isna()) | \
                   ((df_optimal[eff_col] >= MIN_PUMP_EFFICIENCY) & 
                    (df_optimal[eff_col] <= MAX_PUMP_EFFICIENCY))
        
        df_optimal = df_optimal[condition]
        
        # 记录移除数量
        removed = count_before - len(df_optimal)
        if removed > 0:
            pump_removed_counts[i] = removed
            
        # 统计信息
        pump_on_count = pump_on_condition.sum()
        if pump_on_count > 0:
            print(f"  泵{i}: {pump_on_count}条记录中泵开启")
    elif eff_col in df_optimal.columns:
        print(f"注意：泵{i}频率列'{freq_col}'不存在，无法判断泵是否开启，跳过效率筛选")
    else:
        print(f"注意：泵{i}效率列'{eff_col}'不存在")

# 打印各泵筛选结果
if pump_removed_counts:
    print(f"\n各开启泵效率在{MIN_PUMP_EFFICIENCY}-{MAX_PUMP_EFFICIENCY}范围内筛选：")
    for pump_num, count in pump_removed_counts.items():
        print(f"  泵{pump_num}（开启状态）效率超出范围，移除了{count}条记录")
else:
    print(f"\n各开启泵效率在{MIN_PUMP_EFFICIENCY}-{MAX_PUMP_EFFICIENCY}范围内筛选：无记录被移除")

print(f"最终筛选后记录数:", len(df_optimal))

# 检查是否有剩余记录
if len(df_optimal) == 0:
    print("警告：没有满足所有条件的记录！")
    # 创建一个空的DataFrame用于后续处理
    df_optimal = pd.DataFrame()
else:
    # 7. 先排序（在重命名之前，用原始列名排序）
    df_optimal = df_optimal.sort_values(
        by=['总管压力_四舍五入', '总管瞬时流量_四舍五入', '平均吸水井液位_四舍五入', '泵组'],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)

# 8. 重命名列名
column_mapping = {
    '总管压力': 'Header_Press',
    '流量下限': 'Flow_Lower',
    '流量上限': 'Flow_Upper',
    '液位': 'Liquid_Level',
    '泵组': 'Pump_Group',
    
    # 单泵频率给定
    '1号泵频率给定': 'P1_Freq',
    '2号泵频率给定': 'P2_Freq',
    '3号泵频率给定': 'P3_Freq',
    '4号泵频率给定': 'P4_Freq',
    '5号泵频率给定': 'P5_Freq',
    '6号泵频率给定': 'P6_Freq',
    
    # 单泵流量
    '1号泵出口流量': 'P1_Flow',
    '2号泵出口流量': 'P2_Flow',
    '3号泵出口流量': 'P3_Flow',
    '4号泵出口流量': 'P4_Flow',
    '5号泵出口流量': 'P5_Flow',
    '6号泵出口流量': 'P6_Flow',
    
    # 单泵效率
    '1号泵效率': 'P1_Eff',
    '2号泵效率': 'P2_Eff',
    '3号泵效率': 'P3_Eff',
    '4号泵效率': 'P4_Eff',
    '5号泵效率': 'P5_Eff',
    '6号泵效率': 'P6_Eff',
    
    # 其他指标
    '整体效率': 'Eff',
    '千吨水电耗': 'PCTW',
    
    # 排序用的四舍五入列也需要重命名
    '总管压力_四舍五入': 'Header_Press_Rounded',
    '总管瞬时流量_四舍五入': 'Total_Flow_Rounded',
    '平均吸水井液位_四舍五入': 'Liquid_Level_Rounded'
}

# 应用重命名
if len(df_optimal) > 0:
    df_optimal.rename(columns=column_mapping, inplace=True)

# 9. 定义最终输出的列顺序
output_column_order = [
    'Header_Press',           # 总管压力
    'Flow_Lower',             # 流量下限
    'Flow_Upper',             # 流量上限
    'Liquid_Level',           # 液位
    'Pump_Group',             # 泵组
    'P1_Freq', 'P2_Freq', 'P3_Freq', 'P4_Freq', 'P5_Freq', 'P6_Freq',  # 1-6号泵频率
    'P1_Flow', 'P2_Flow', 'P3_Flow', 'P4_Flow', 'P5_Flow', 'P6_Flow',  # 1-6号泵流量
    'Total_Flow',             # 六泵总出口流量
    'Flow_Deviation',         # 流量偏差
    'P1_Eff', 'P2_Eff', 'P3_Eff', 'P4_Eff', 'P5_Eff', 'P6_Eff',  # 1-6号泵效率
    'Eff',                    # 整体效率
    'PCTW'                    # 千吨水电耗
]

# 10. 创建最终结果DataFrame
if len(df_optimal) > 0:
    # 检查DataFrame中已有的列
    available_columns = [col for col in output_column_order if col in df_optimal.columns]
    missing_columns = [col for col in output_column_order if col not in df_optimal.columns]
    
    if missing_columns:
        print(f"\n警告：以下列不存在，将创建为NaN列: {missing_columns}")
        for col in missing_columns:
            if col not in df_optimal.columns:
                df_optimal[col] = 0
    
    # 重新排列列的顺序
    df_results = df_optimal[output_column_order].copy()
    
else:
    # 如果没有符合条件的记录，创建一个空的DataFrame
    df_results = pd.DataFrame(columns=output_column_order)
    print("\n注意：没有符合条件的记录，创建空结果DataFrame")

# 11. 输出结果
print("\n" + "=" * 80)
print("最终筛选条件：")
print(f"- 整体效率(Eff)范围: {MIN_OVERALL_EFFICIENCY} - {MAX_OVERALL_EFFICIENCY}")
print(f"- 运行泵效率(P1_Eff~P6_Eff)范围: {MIN_PUMP_EFFICIENCY} - {MAX_PUMP_EFFICIENCY} (仅对开启泵)")
print(f"- 泵开启阈值: 频率 > {PUMP_ON_THRESHOLD}")
print(f"- 千吨水电耗(PCTW)范围: {MIN_PCTW} - {MAX_PCTW}")
print("=" * 80)

if len(df_results) > 0:
    print(f"\n处理完成！找到 {len(df_results)} 条满足所有条件的最优解记录。")
    print("=" * 80)
    print("\n前5行数据：")
    print(df_results.head())
    
    print("\n" + "=" * 80)
    print("列名及其顺序：")
    for i, col in enumerate(df_results.columns, 1):
        print(f"{i:2d}. {col}")
    
    print("\n" + "=" * 80)
    print("数据统计信息：")
    print(f"记录数量: {len(df_results)}")
    print(f"泵组组合数: {df_results['Pump_Group'].nunique()}")
    
    # 统计各参数的取值范围
    if 'Eff' in df_results.columns:
        eff_data = df_results['Eff'].round(2)
        print(f"\n整体效率(Eff)统计:")
        print(f"  范围: {eff_data.min():.3f} - {eff_data.max():.3f}")
        print(f"  平均值: {eff_data.mean():.3f}")
        print(f"  中位数: {eff_data.median():.3f}")
    
    if 'PCTW' in df_results.columns:
        pctw_data = df_results['PCTW'].round(2)
        print(f"\n千吨水电耗(PCTW)统计:")
        print(f"  范围: {pctw_data.min():.2f} - {pctw_data.max():.2f}")
        print(f"  平均值: {pctw_data.mean():.2f}")
        print(f"  中位数: {pctw_data.median():.2f}")
    
    # 各泵效率统计（仅统计开启的泵）
    print(f"\n运行泵效率统计(仅P1_Eff~P6_Eff中开启的泵):")
    for i in range(1, 7):
        freq_col = f'P{i}_Freq'
        eff_col = f'P{i}_Eff'
        
        if freq_col in df_results.columns and eff_col in df_results.columns:
            # 找出开启的泵
            pump_on_mask = df_results[freq_col] > PUMP_ON_THRESHOLD
            pump_on_data = df_results.loc[pump_on_mask, eff_col].dropna()
            
            if len(pump_on_data) > 0:
                print(f"  泵{i}(开启): 范围={pump_on_data.min():.3f}-{pump_on_data.max():.3f}, "
                      f"平均={pump_on_data.mean():.3f}, 开启次数={len(pump_on_data)}")
            else:
                print(f"  泵{i}: 无开启记录")
    
    # 保存为CSV文件
    output_file = '历史最优策略.csv'
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("\n" + "=" * 80)
    print(f"结果已保存到 {output_file}")
    print("=" * 80)
    
    # 添加汇总统计
    original_count = len(df)
    filtered_count = len(df_results)
    if original_count > 0:
        percentage = (filtered_count / original_count) * 100
        print(f"\n汇总信息:")
        print(f"  原始数据: {original_count} 条记录")
        print(f"  筛选后数据: {filtered_count} 条记录")
        print(f"  保留比例: {percentage:.2f}%")
    print("=" * 80)
else:
    print("没有找到满足所有条件的记录。")