import pandas as pd
import numpy as np
import openpyxl
from openpyxl.styles import PatternFill
import os
from datetime import datetime, timedelta
def match_and_compare_csv(file1_path, file2_path, output_path, file2_only_data_path):
    """
    匹配两个CSV文件数据，按总管效率比较，并添加详细统计信息
    """
    # ===================== 1. 读取数据 =====================
    print("正在读取数据文件...")
    df1 = pd.read_csv(file1_path, encoding='utf-8-sig')
    df2 = pd.read_csv(file2_path, encoding='utf-8-sig')
    
    # ===================== 2. 定义匹配列 =====================
    # 使用指定的5个列进行数据对齐
    match_columns = ['Header_Press', 'Flow_Lower', 'Flow_Upper', 'Liquid_Level', 'Pump_Group']
    
    # 检查匹配列是否存在
    for col in match_columns:
        if col not in df1.columns:
            raise ValueError(f"File1中缺少匹配列: {col}")
        if col not in df2.columns:
            raise ValueError(f"File2中缺少匹配列: {col}")
    
    # ===================== 3. 准备匹配键 =====================
    print("准备匹配键并标准化格式...")
    
    # 为file1和file2准备匹配键
    for df in [df1, df2]:
        # 确保数据类型一致
        df['match_key_压力'] = df['Header_Press'].round(2)
        df['match_key_流量下限'] = df['Flow_Lower'].round(2)
        df['match_key_流量上限'] = df['Flow_Upper'].round(2)
        df['match_key_液位'] = df['Liquid_Level'].round(1)
        df['match_key_组合'] = df['Pump_Group'].astype(str).str.strip()
        
        # 构建完整的匹配键
        df['full_match_key'] = df.apply(
            lambda x: (x['match_key_压力'], x['match_key_流量下限'], x['match_key_流量上限'], 
                       x['match_key_液位'], x['match_key_组合']), axis=1)
    
    # ===================== 4. 查找关键列 =====================
    def find_efficiency_col(df):
        """查找总管效率列"""
        if 'Eff' in df.columns:
            return df.columns.get_loc('Eff')
        raise ValueError("未找到总管效率列(Eff)")
    
    def find_kwh_col(df):
        """查找千吨水电耗列"""
        if 'PCTW' in df.columns:
            return df.columns.get_loc('PCTW')
        return None
    
    def find_pump_freq_cols(df):
        """查找泵频率给定列"""
        pump_freq_cols = []
        for i in range(1, 7):
            col_name = f'P{i}_Freq'
            if col_name in df.columns:
                pump_freq_cols.append(df.columns.get_loc(col_name))
        return pump_freq_cols
    
    df1_eff_idx = find_efficiency_col(df1)
    df2_eff_idx = find_efficiency_col(df2)
    df1_kwh_idx = find_kwh_col(df1)
    df2_kwh_idx = find_kwh_col(df2)
    df1_pump_freq_idxs = find_pump_freq_cols(df1)
    df2_pump_freq_idxs = find_pump_freq_cols(df2)
    
    # ===================== 5. 定义输出列 =====================
    # 使用完整的列列表
    output_columns = [
        'Header_Press', 'Flow_Lower', 'Flow_Upper', 'Liquid_Level', 'Pump_Group',
        'P1_Freq', 'P2_Freq', 'P3_Freq', 'P4_Freq', 'P5_Freq', 'P6_Freq',
        'P1_Flow', 'P2_Flow', 'P3_Flow', 'P4_Flow', 'P5_Flow', 'P6_Flow',
        'Total_Flow', 'Flow_Deviation',
        'P1_Eff', 'P2_Eff', 'P3_Eff', 'P4_Eff', 'P5_Eff', 'P6_Eff',
        'Eff', 'PCTW'
    ]
    
    # 为file2构建匹配键到行数据的映射 [1,3](@ref)
    df2_match_map = {}
    for idx, row in df2.iterrows():
        match_key = row['full_match_key']
        if match_key not in df2_match_map:
            row_output = {}
            for col in output_columns:
                row_output[col] = row[col] if col in df2.columns else None
            df2_match_map[match_key] = {
                'row_data': row_output,
                'efficiency': row.iloc[df2_eff_idx],
                'kwh_per_kt': row.iloc[df2_kwh_idx] if df2_kwh_idx is not None else None,
                'pump_freqs': [row.iloc[i] for i in df2_pump_freq_idxs] if df2_pump_freq_idxs else []
            }
    
    df1_match_keys = set(df1['full_match_key'])
    df2_match_keys = set(df2['full_match_key'])
    
    # ===================== 6. 合并匹配的数据行 =====================
    print("合并匹配的数据行...")
    merged_data = []
    color_flags = []
    
    # 统计变量
    file1_only_count = 0
    file2_only_count = 0
    identical_count = 0
    file1_better_count = 0
    file2_better_count = 0
    
    # File1效率更高时的统计
    file1_better_eff_improvements = []
    file1_better_kwh_reductions = []
    
    # File2效率更高时的统计  
    file2_better_eff_improvements = []
    file2_better_kwh_reductions = []
    
    # 处理file1中的所有行 [1,3](@ref)
    for idx, row1 in df1.iterrows():
        match_key = row1['full_match_key']
        row1_eff = row1.iloc[df1_eff_idx]
        row1_kwh = row1.iloc[df1_kwh_idx] if df1_kwh_idx is not None else None
        row1_pump_freqs = [row1.iloc[i] for i in df1_pump_freq_idxs] if df1_pump_freq_idxs else []
        
        # 准备合并后的行数据
        merged_row = {}
        
        # 添加file1的数据（前N列）
        for col in output_columns:
            col_name = f'file1_{col}'
            merged_row[col_name] = row1[col] if col in df1.columns else None
        
        if match_key in df2_match_map:
            file2_item = df2_match_map[match_key]
            row2_eff = file2_item['efficiency']
            row2_kwh = file2_item['kwh_per_kt']
            row2_pump_freqs = file2_item['pump_freqs']
            
            # 添加file2的数据（后N列）
            for col in output_columns:
                col_name = f'file2_{col}'
                merged_row[col_name] = file2_item['row_data'][col] if file2_item['row_data'][col] is not None else None
            
            # 检查所有关键字段是否完全相同
            is_identical = (
                all(abs(f1 - f2) < 0.001 if f1 is not None and f2 is not None else f1 == f2 
                    for f1, f2 in zip(row1_pump_freqs, row2_pump_freqs))  # 泵频率基本相同
            )
            
            if is_identical:
                # 数据完全相同 - 灰色
                color_flags.append('gray')
                identical_count += 1
            else:
                # 计算效率差异
                eff_improvement = row1_eff - row2_eff
                
                if row1_eff >= row2_eff:
                    # File1效率更高 - 浅绿色
                    color_flags.append('light_green')
                    file1_better_count += 1
                    file1_better_eff_improvements.append(eff_improvement)
                    
                    # 计算能耗降低（如果存在能耗数据）
                    if row1_kwh is not None and row2_kwh is not None:
                        kwh_reduction = row2_kwh - row1_kwh
                        file1_better_kwh_reductions.append(kwh_reduction)
                else:
                    # File2效率更高 - 黄色
                    color_flags.append('yellow')
                    file2_better_count += 1
                    file2_better_eff_improvements.append(abs(eff_improvement))
                    
                    # 计算能耗降低（如果存在能耗数据）
                    if row1_kwh is not None and row2_kwh is not None:
                        kwh_reduction = row1_kwh - row2_kwh
                        file2_better_kwh_reductions.append(kwh_reduction)
        else:
            # 只有file1数据 - 白色
            for col in output_columns:
                col_name = f'file2_{col}'
                merged_row[col_name] = None
            color_flags.append('white')
            file1_only_count += 1
        
        merged_data.append(merged_row)
    
    # 处理只有file2数据的行 [1,3](@ref)
    file2_only_rows = []
    for match_key in df2_match_keys:
        if match_key not in df1_match_keys:
            file2_item = df2_match_map[match_key]
            merged_row = {}
            
            # file1数据为空（前N列）
            for col in output_columns:
                col_name = f'file1_{col}'
                merged_row[col_name] = None
            
            # 添加file2的数据（后N列）
            for col in output_columns:
                col_name = f'file2_{col}'
                merged_row[col_name] = file2_item['row_data'][col] if file2_item['row_data'][col] is not None else None
            
            merged_data.append(merged_row)
            color_flags.append('orange')
            file2_only_count += 1
            file2_only_rows.append(merged_row)
    
    # ===================== 7. 创建结果DataFrame =====================
    print("创建结果DataFrame...")
    
    # 创建有序的列列表
    ordered_columns = []
    for col in output_columns:
        ordered_columns.append(f'file1_{col}')
    for col in output_columns:
        ordered_columns.append(f'file2_{col}')
    
    # 创建DataFrame并填充NaN值
    df_result = pd.DataFrame(merged_data)
    
    # 确保所有需要的列都存在
    for col in ordered_columns:
        if col not in df_result.columns:
            df_result[col] = None
    
    # 重新排列列顺序并填充NaN
    df_result = df_result[ordered_columns]
    df_result = df_result.fillna('N/A')
    
    # ===================== 8. 计算详细统计信息 =====================
    def calculate_detailed_stats(data_list, data_name, unit=''):
        """计算详细的统计指标"""
        if not data_list:
            print(f"  {data_name}: 无数据")
            return None
        
        arr = np.array(data_list)
        stats = {
            'count': len(arr),
            'mean': np.mean(arr),
            'median': np.median(arr),
            'variance': np.var(arr, ddof=1),
            'std': np.std(arr, ddof=1),
            'max': np.max(arr),
            'min': np.min(arr),
            'unit': unit
        }
        
        print(f"  {data_name}统计（共{stats['count']}行）：")
        print(f"    平均数：{stats['mean']:.4f}{unit}")
        print(f"    中位数：{stats['median']:.4f}{unit}")
        print(f"    方差：{stats['variance']:.4f}")
        print(f"    标准差：{stats['std']:.4f}")
        print(f"    最大值：{stats['max']:.4f}{unit}")
        print(f"    最小值：{stats['min']:.4f}{unit}")
        print()
        
        return stats
    
    # ===================== 9. 打印详细统计结果 =====================
    total_rows = len(df_result)
    print("\n" + "="*60)
    print("详细统计信息")
    print("="*60)
    
    def print_count_with_percentage(count, description):
        percentage = (count / total_rows) * 100 if total_rows > 0 else 0
        print(f"{description}：{count} ({percentage:.2f}%)")
    
    print_count_with_percentage(file1_only_count, "File1独有数据行数")
    print_count_with_percentage(file2_only_count, "File2独有数据行数")
    print_count_with_percentage(identical_count, "完全相同数据行数")
    print_count_with_percentage(file1_better_count, "File1效率更高行数")
    print_count_with_percentage(file2_better_count, "File2效率更高行数")
    
    # 效率对比统计
    print("File1效率更高时：")
    file1_eff_stats = calculate_detailed_stats(file1_better_eff_improvements, "效率提升", "")
    file1_kwh_stats = calculate_detailed_stats(file1_better_kwh_reductions, "千吨水电耗降低", "kWh/kt")
    
    print("File2效率更高时：")
    file2_eff_stats = calculate_detailed_stats(file2_better_eff_improvements, "效率提升", "")
    file2_kwh_stats = calculate_detailed_stats(file2_better_kwh_reductions, "千吨水电耗降低", "kWh/kt")
    
    # ===================== 10. 输出带颜色的Excel文件 =====================
    print(f"正在生成带颜色标记的结果文件：{output_path}")
    
    # 颜色映射定义 [11](@ref)
    color_mapping = {
        'light_green': PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid"),  # File1效率更高
        'yellow': PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid"),      # File2效率更高
        'white': PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid"),       # 只有File1数据
        'orange': PatternFill(start_color="FFA500", end_color="FFA500", fill_type="solid"),      # 只有File2数据
        'gray': PatternFill(start_color="808080", end_color="808080", fill_type="solid")         # 数据完全相同
    }
    
    # 使用ExcelWriter输出带颜色的Excel文件 [11](@ref)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_result.to_excel(writer, sheet_name='对比结果', index=False)
        
        # 获取工作表并应用颜色
        ws = writer.sheets['对比结果']
        for row_idx, color in enumerate(color_flags, start=2):  # 从第2行开始（跳过标题行）
            for col_idx in range(1, len(ordered_columns) + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.fill = color_mapping[color]
    
    # ===================== 11. 输出file2独有数据到新Excel文件 =====================
    if file2_only_rows:
        print(f"正在生成file2独有数据文件：{file2_only_data_path}")
        df_file2_only = pd.DataFrame(file2_only_rows)
        
        # 确保所有列都存在
        for col in ordered_columns:
            if col not in df_file2_only.columns:
                df_file2_only[col] = 'N/A'
        
        df_file2_only = df_file2_only[ordered_columns]
        df_file2_only.to_excel(file2_only_data_path, index=False)
    
    # ===================== 12. 返回完整统计信息 =====================
    stats_summary = {
        'total_rows': total_rows,
        'file1_only_count': file1_only_count,
        'file2_only_count': file2_only_count,
        'identical_count': identical_count,
        'file1_better_count': file1_better_count,
        'file2_better_count': file2_better_count,
        'file1_better_eff_stats': file1_eff_stats,
        'file1_better_kwh_stats': file1_kwh_stats,
        'file2_better_eff_stats': file2_eff_stats,
        'file2_better_kwh_stats': file2_kwh_stats
    }
    
    return stats_summary

if __name__ == "__main__":

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    FILE1_PATH = r"D:\MachineView\pump_optimization_ga_20251231_154226.csv"
    FILE2_PATH = "历史最优策略.csv"
    OUTPUT_PATH = f"对比结果_{time_str}.xlsx"
    FILE2_ONLY_DATA_PATH = f"历史策略独有数据_{time_str}.xlsx"
    
    # 检查文件是否存在
    for file_path in [FILE1_PATH, FILE2_PATH]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")
    
    try:
        stats = match_and_compare_csv(FILE1_PATH, FILE2_PATH, OUTPUT_PATH, FILE2_ONLY_DATA_PATH)
        print("\n" + "="*60)
        print("处理完成！")
        print("="*60)
        print(f"总行数：{stats['total_rows']}")
        
        def print_result_with_percentage(count, description):
            percentage = (count / stats['total_rows']) * 100 if stats['total_rows'] > 0 else 0
            print(f"{description}：{count} ({percentage:.2f}%)")
        
        print_result_with_percentage(stats['file1_only_count'], "File1独有数据行数")
        print_result_with_percentage(stats['file2_only_count'], "File2独有数据行数")
        print_result_with_percentage(stats['identical_count'], "完全相同数据行数")
        print_result_with_percentage(stats['file1_better_count'], "File1效率更高行数")
        print_result_with_percentage(stats['file2_better_count'], "File2效率更高行数")
        print(f"结果文件已保存至：{OUTPUT_PATH}")
        if stats['file2_only_count'] > 0:
            print(f"File2独有数据已保存至：{FILE2_ONLY_DATA_PATH}")
    except Exception as e:
        print(f"❌ 程序执行出错：{str(e)}")
        import traceback
        traceback.print_exc()