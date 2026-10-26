import pandas as pd
import numpy as np
import time
from tqdm import tqdm

# 物理常量
DENSITY_WATER = 1000  # 水的密度 kg/m³
GRAVITY = 9.81        # 重力加速度 m/s²

def round_to_nearest(value, base):
    """四舍五入到指定基数的倍数"""
    if pd.isna(value):
        return value
    else:
        if round(value / base) * base > value:
            return round(value / base) * base - base
        else:
            return round(value / base) * base

def calculate_pump_efficiency(input_power, flow, head):
    """
    计算单个泵的效率（不转成百分比）
    效率 = (输出功率 / 输入功率)
    输出功率 = ρgQH/3600000 (kW)
    """
    # 检查输入有效性
    if pd.isna(input_power) or pd.isna(flow) or pd.isna(head):
        return 0.0
    
    if input_power <= 0 or flow <= 0 or head <= 0:
        return 0.0
    
    try:
        # 计算输出功率 (kW) = ρ * g * Q * H / 3600000
        output_power = (DENSITY_WATER * GRAVITY * flow * head) / 3600000
        
        # 计算效率 = 输出功率 / 输入功率（不转百分比）
        efficiency = output_power / input_power
        
        # 限制效率在合理范围内
        return max(0.0, min(efficiency, 1.0))
    except:
        return 0.0

def calculate_operating_condition(row):
    """根据泵运行状态生成工况字符串（6位）"""
    condition = ""
    for i in range(1, 7):
        run_col = f'泵运行{i}'
        if run_col in row:
            # 运行状态为1表示运行，其他表示停止
            status = int(row[run_col]) if not pd.isna(row[run_col]) else 0
            condition += "1" if status == 1 else "0"
        else:
            condition += "0"
    return condition

def process_pump_data(input_file, output_file="processed_pump_data.csv"):
    """
    主处理函数
    """
    print("=" * 60)
    print("📊 泵数据处理程序启动")
    print("=" * 60)
    
    # 1. 读取数据
    print("📁 正在读取数据文件...")
    start_time = time.time()
    
    try:
        # 尝试不同的编码方式
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(input_file, encoding=encoding)
                print(f"✅ 使用 {encoding} 编码读取成功")
                print(f"📈 原始数据: {len(df)} 行, {len(df.columns)} 列")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print("❌ 无法读取文件，请检查文件路径和编码")
            return
            
    except FileNotFoundError:
        print(f"❌ 文件未找到: {input_file}")
        return
    
    read_time = time.time() - start_time
    print(f"⏱️  读取用时: {read_time:.2f}秒")
    
    # 创建进度条
    print("\n🔄 开始数据处理...")
    pbar = tqdm(total=100, desc="处理进度", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    
    # 新增步骤1：将泵频率给定与泵运行状态相乘
    pbar.update(10)
    print("\n🔧 正在处理泵频率给定与运行状态相乘...")
    
    # 检查并处理每个泵的频率给定
    for i in range(1, 7):
        freq_col = f'泵频率给定{i}'
        run_col = f'泵运行{i}'
        flow_col = f'泵出口流量{i}'
        head_col = f'泵{i}_扬程'
        # 检查频率给定列是否存在
        if freq_col in df.columns:
            # 检查运行状态列是否存在
            if run_col in df.columns:
                # 将泵频率给定与泵运行状态相乘（运行状态为1或0）
                # 使用loc确保避免SettingWithCopyWarning
                df.loc[:, freq_col] = df[freq_col] * df[run_col]
                df.loc[:, flow_col] = df[flow_col] * df[run_col]
                df.loc[:, head_col] = df[head_col] * df[run_col]
                print(f"✅ 已处理泵{i}: 频率给定 × 运行状态")
            else:
                print(f"⚠️ 泵{i}运行状态列'{run_col}'不存在，跳过相乘操作")
        else:
            print(f"⚠️ 泵{i}频率给定列'{freq_col}'不存在")
    
    # 新增步骤2：删除至少有一个泵的频率在[30, 40)之间的数据行
    pbar.update(10)
    print("\n🗑️  正在删除泵频率在[30, 40)之间的数据行...")
    original_count = len(df)
    
    # 定义泵频率列名
    pump_freq_columns = [f'泵频率给定{i}' for i in range(1, 7)]
    
    # 检查哪些频率列存在
    existing_freq_columns = [col for col in pump_freq_columns if col in df.columns]
    
    if existing_freq_columns:
        print(f"🔍 找到频率列: {existing_freq_columns}")
        
        # 创建筛选条件：所有泵的频率都不在[30, 40)之间
        # 即对于每个存在的频率列，频率要么<30，要么>=40，要么为NaN/0
        filter_condition = True  # 初始化为True（保留所有行）
        
        for freq_col in existing_freq_columns:
            # 条件：该泵频率不在[30, 40)之间
            # 使用 | 运算符组合多个条件
            col_condition = (
                (df[freq_col] < 30) |    # 频率小于30Hz
                (df[freq_col] >= 40) |   # 频率大于等于40Hz
                (df[freq_col].isna()) |  # 频率为NaN
                (df[freq_col] == 0)      # 频率为0（泵停止）
            )
            filter_condition &= col_condition  # 所有条件都必须满足
        
        # 应用筛选条件
        df = df[filter_condition]
        
        removed_count = original_count - len(df)
        print(f"✅ 已删除 {removed_count} 行包含泵频率在[30, 40)之间的数据")
        print(f"📊 剩余数据: {len(df)} 行")
        
        if len(df) == 0:
            print("❌ 警告：筛选后没有剩余数据！")
            return
    else:
        print("⚠️ 未找到任何泵频率给定列，跳过频率筛选")
    
    # 2. 计算平均吸水井液位（原步骤2，现在变为步骤3）
    pbar.update(10)
    if '一期吸水井液位' in df.columns and '二期吸水井液位' in df.columns:
        df['平均吸水井液位_原始'] = (df['一期吸水井液位'] + df['二期吸水井液位']) / 2
        print("✅ 已计算平均吸水井液位（原始）")
    else:
        print("⚠️ 未找到吸水井液位列，将使用默认值")
        df['平均吸水井液位_原始'] = 5.0  # 默认值
    
    # 3. 计算四舍五入后的值（原步骤3，现在变为步骤4）
    pbar.update(20)
    
    # 总管瞬时流量 - 四舍五入到最近的50的倍数
    if '总管瞬时流量' in df.columns:
        df['总管瞬时流量_四舍五入'] = df['总管瞬时流量'].apply(lambda x: round_to_nearest(x, 50))
        print("✅ 已计算四舍五入后的总管瞬时流量")
    
    # 总管压力 - 四舍五入到小数点后2位
    if '总管压力' in df.columns:
        df['总管压力_四舍五入'] = df['总管压力'].apply(lambda x: round(x, 2) if not pd.isna(x) else x)
        print("✅ 已计算四舍五入后的总管压力")
    
    # 平均吸水井液位 - 四舍五入到最近的0.5的倍数
    df['平均吸水井液位_四舍五入'] = df['平均吸水井液位_原始'].apply(lambda x: round_to_nearest(x, 0.5))
    print("✅ 已计算四舍五入后的平均吸水井液位")
    
    # 4. 生成运行组合（原步骤4，现在变为步骤5）
    pbar.update(20)
    print("🔧 正在生成运行组合...")
    df['泵组'] = df.apply(calculate_operating_condition, axis=1)
    print(f"✅ 已生成运行组合，共有 {df['泵组'].nunique()} 种不同组合")
    
    # 5. 计算各泵效率（不转百分比）（原步骤5，现在变为步骤6）
    pbar.update(30)
    print("\n⚡ 正在计算各泵效率...")
    
    for i in range(1, 7):
        input_power_col = f'泵{i}_输入功率_kW'
        flow_col = f'泵出口流量{i}'
        head_col = f'泵{i}_扬程'
        freq_col = f'泵频率给定{i}'
        
        # 检查是否有计算效率所需的数据
        if all(col in df.columns for col in [input_power_col, flow_col, head_col]):
            df[f'{i}号泵效率'] = df.apply(
                lambda row: calculate_pump_efficiency(
                    row[input_power_col], 
                    row[flow_col], 
                    row[head_col]
                ), 
                axis=1
            )
            print(f"✅ 已计算泵{i}效率")
        else:
            df[f'{i}号泵效率'] = 0.0
            print(f"⚠️ 泵{i}数据不完整，效率设为0")
            
        # 获取频率值（已经在前面的步骤中处理过）
        if freq_col in df.columns:
            df[f'{i}号泵频率给定'] = df[freq_col]
        else:
            df[f'{i}号泵频率给定'] = 0.0
            print(f"⚠️ 未找到泵{i}频率给定列")
    
    # 6. 计算流量下限和上限（原步骤6，现在变为步骤7）
    pbar.update(10)
    if '总管瞬时流量' in df.columns:
        # 如果没有四舍五入列，直接从原始列计算
        df['流量下限'] = df['总管瞬时流量'].apply(lambda x: round_to_nearest(x, 50))
        df['流量上限'] = df['流量下限'] + 50
        print("✅ 已计算流量下限和上限")
    else:
        df['流量下限'] = 0
        df['流量上限'] = 0
        print("⚠️ 未找到总管瞬时流量列，流量上下限设为0")
    
    # 7. 准备输出数据（按照中文列名格式）（原步骤7，现在变为步骤8）
    pbar.update(10)
    print("\n📝 准备输出数据...")
    
    # 创建输出DataFrame
    output_data = {}
    
    # 按照中文列名顺序添加列
    # 1. 总管压力 - 四舍五入后
    if '总管压力_四舍五入' in df.columns:
        output_data['总管压力'] = df['总管压力_四舍五入']
    elif '总管压力' in df.columns:
        output_data['总管压力'] = df['总管压力'].apply(lambda x: round(x, 2) if not pd.isna(x) else x)
    else:
        output_data['总管压力'] = 0.0
    
    # 2. 流量下限
    if '流量下限' in df.columns:
        output_data['流量下限'] = df['流量下限']
    
    # 3. 流量上限
    if '流量上限' in df.columns:
        output_data['流量上限'] = df['流量上限']
    
    # 4. 液位 - 四舍五入后
    output_data['液位'] = df['平均吸水井液位_四舍五入']
    
    # 5. 泵组 - 运行组合
    output_data['泵组'] = df['泵组']
    
    # 6. 1-6号泵的给定频率（已经处理过）
    for i in range(1, 7):
        output_data[f'{i}号泵频率给定'] = df[f'{i}号泵频率给定']
    
    # 7. 1-6号泵的运行效率（不转百分比）
    for i in range(1, 7):
        output_data[f'{i}号泵效率'] = df[f'{i}号泵效率']
    
    # 8. 整体效率
    if '总管_效率' in df.columns:
        output_data['整体效率'] = df['总管_效率'].apply(lambda x: round(x, 2) if not pd.isna(x) else x)
    else:
        output_data['整体效率'] = 0.0
    
    # 9. 千吨水电耗
    if '千吨水电耗_kWh_kt' in df.columns:
        output_data['千吨水电耗'] = df['千吨水电耗_kWh_kt'].apply(lambda x: round(x, 2) if not pd.isna(x) else x)
    else:
        output_data['千吨水电耗'] = 0.0
    
    # 创建输出DataFrame
    output_df = pd.DataFrame(output_data)
    
    # 添加原始数据和四舍五入后的数据（用于参考）
    if '总管瞬时流量' in df.columns:
        output_df['总管瞬时流量_原始'] = df['总管瞬时流量']
    if '总管瞬时流量_四舍五入' in df.columns:
        output_df['总管瞬时流量_四舍五入'] = df['总管瞬时流量_四舍五入']
    
    if '总管压力' in df.columns:
        output_df['总管压力_原始'] = df['总管压力']
    if '总管压力_四舍五入' in df.columns:
        output_df['总管压力_四舍五入'] = df['总管压力_四舍五入']
    
    if '平均吸水井液位_原始' in df.columns:
        output_df['平均吸水井液位_原始'] = df['平均吸水井液位_原始']
    if '平均吸水井液位_四舍五入' in df.columns:
        output_df['平均吸水井液位_四舍五入'] = df['平均吸水井液位_四舍五入']
    
    # 添加各泵的出口流量和扬程（额外的信息）
    for i in range(1, 7):
        flow_col = f'泵出口流量{i}'
        head_col = f'泵{i}_扬程'
        if flow_col in df.columns:
            output_df[f'{i}号泵出口流量'] = df[flow_col]
        if head_col in df.columns:
            output_df[f'{i}号泵扬程'] = df[head_col]
    
    # 保存结果
    pbar.update(10)
    print("\n💾 正在保存处理结果...")
    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    total_time = time.time() - start_time
    pbar.close()
    
    print("\n" + "=" * 60)
    print("✅ 数据处理完成！")
    print("=" * 60)
    print(f"📊 处理结果:")
    print(f"   总数据行数: {len(output_df)}")
    print(f"   输出列数: {len(output_df.columns)}")
    print(f"   总处理时间: {total_time:.2f}秒")
    print(f"   输出文件: {output_file}")
    
    # 显示所有列名
    print("\n📋 输出文件包含以下列:")
    columns = output_df.columns.tolist()
    for i, col in enumerate(columns, 1):
        print(f"   {i:2d}. {col}")
    
    return output_df

# 运行主程序
if __name__ == "__main__":
    # 设置输入文件路径
    input_file = "cleaned_pump_data_iqr.csv"  # 请修改为您的实际文件名
    
    print("🏭 泵数据处理程序")
    print("功能说明:")
    print("1. 将泵频率给定与泵运行状态相乘")
    print("2. 删除至少有一个泵的频率在[30, 40)之间的数据行")
    print("3. 计算每个泵的效率（输出功率/输入功率，不转百分比）")
    print("4. 计算平均吸水井液位")
    print("5. 计算四舍五入后的总管流量、压力、液位")
    print("6. 生成泵运行组合（如110000）")
    print("7. 计算流量下限和上限（流量下限=四舍五入后的总管瞬时流量，上限=四舍五入后的总管瞬时流量+50）")
    print("8. 输出中文列名的数据表")
    print("=" * 60)
    
    # 运行处理程序
    result_df = process_pump_data(input_file)
    
    if result_df is not None:
        # 显示前几行数据
        print("\n📄 数据预览（前5行）:")
        preview_columns = ['总管压力', '流量下限', '流量上限', '液位', '泵组', '整体效率', '千吨水电耗']
        existing_columns = [col for col in preview_columns if col in result_df.columns]
        print(result_df[existing_columns].head())