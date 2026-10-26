import os
import pandas as pd
import numpy as np
from itertools import product
import time
from tqdm import tqdm

# ====== 常量定义 ======
DENSITY_WATER = 1000  # kg/m³
GRAVITY = 9.81        # m/s²
PRESSURE_TO_HEAD = 102  # 1MPa = 102m扬程
EFFICIENCY_DENOMINATOR = 3600000  # 输出功率计算分母
FLOW_EPS = 0  # 流量极小值阈值
POWER_EPS = 0  # 功率极小值阈值

# 泵额定功率 (kW) 和额定流量 (m³/h)
PUMP_RATED_POWER = np.array([315, 315, 315, 315, 710])
PUMP_RATED_FLOWS = np.array([1460, 1460, 1460, 1460, 2910])
POWER_FACTORS = [0.89, 0.89, 0.86, 0.89, 0.83]

# 优化参数
VALID_FREQUENCIES = list(np.arange(40, 50.5, 1))  # 40-50Hz，步长1

MIN_KWH_PER_KT = 100.0
MAX_KWH_PER_KT = 200.0

MIN_TOTAL_EFFICIENCY = 0.4  # 总管效率下限
MAX_TOTAL_EFFICIENCY = 0.9  # 总管效率上限

MIN_PUMP_EFFICIENCY = 0.4  # 单泵效率下限
MAX_PUMP_EFFICIENCY = 0.9  # 单泵效率上限

# 泵流量函数系数
PUMP_3D_FLOW_FUNC = {
    0: {'coeffs': [7483.352305548825, 792.841314929868, -131587.371578784921, -10.424955986413, 
                   268743.490156018001, -344.090414351132, -0.004235290134, -159510.904414695862, 
                   13.264406475426, -678.727629257450]},
    1: {'coeffs': [-150328.878871415509, 10383.495654498884, -3277.728371870372, -256.582821735820, 
                   -219277.254041541310, 4236.024690479263, 2.194183172041, 44971.651086584083, 
                   -77.943309066847, 3405.620839865252]},
    2: {'coeffs': [-18777.664737047613, 1990.071514525810, -69201.479254340113, -52.300842202801, 
                   49685.449839157474, 1828.015899956583, 0.501073012145, -73705.617737370208, 
                   -32.065437438840, 1202.923910141764]},
    3: {'coeffs': [-126667.211010583997, 7418.266794159802, 111852.703528970058, -207.808211499261, 
                   -646714.132205531001, 8190.014808920248, 1.787357985788, 530297.221352579538, 
                   -67.940493217905, -2141.077162151781]},
    4: {'coeffs': [10390.778782267014, 110.508642523158, -88133.619814955993, 13.784264623553, 
                   259882.698229957081, -1960.086924638748, -0.353768665379, -75652.394218882531, 
                   64.288311711990, -3643.743212249265]}
}

def format_pump_group(pump_group_str):
    """
    格式化泵组字符串：在第一个"1"前补0至6位
    示例：
    "1100" → "001100"（开3、4泵）
    "11100" → "011100"（开2、3、4泵）
    "101000" → "101000"（已6位，无需补）
    """
    pump_group_str = str(pump_group_str).strip()
    if not pump_group_str or not pump_group_str.isdigit():
        return "000000"
    
    first_one_idx = pump_group_str.find("1")
    if first_one_idx == -1:
        return "000000"
    
    need_pad = 6 - len(pump_group_str)
    if need_pad <= 0:
        return pump_group_str[:6]
    
    padded_str = "0" * need_pad + pump_group_str
    return padded_str[:6]

def calculate_pump_flow_3d(pump_idx, frequency, actual_pressure):
    """基于三维多项式函数计算单个泵的出口流量"""
    if frequency <= FLOW_EPS:
        return 0.0

    coeffs = PUMP_3D_FLOW_FUNC[pump_idx]['coeffs']
    f, p = frequency, actual_pressure

    f2, f3 = f**2, f**3
    p2, p3 = p**2, p**3

    flow = (coeffs[0] + coeffs[1]*f + coeffs[2]*p + coeffs[3]*f2 + coeffs[4]*p2 +
            coeffs[5]*f*p + coeffs[6]*f3 + coeffs[7]*p3 + coeffs[8]*f2*p + coeffs[9]*f*p2)
    return max(flow, 0.0)

def calculate_pump_input_power(pump_idx, frequency):
    """计算单个泵的输入功率"""
    if frequency <= POWER_EPS:
        return 0.0
    
    rated_power = PUMP_RATED_POWER[pump_idx]
    power_factor = POWER_FACTORS[pump_idx]
    base_power = rated_power * (frequency / 50) ** 3
    return base_power * power_factor

def calculate_single_pump_output_power(pump_flow, total_pressure, water_level):
    """计算单个泵的输出功率"""
    if pump_flow <= FLOW_EPS:
        return 0.0

    H = total_pressure * PRESSURE_TO_HEAD
    effective_head = H - (water_level - 2.35)
    if effective_head < 0:
        return 0.0

    output_power = (DENSITY_WATER * GRAVITY * pump_flow * effective_head) / EFFICIENCY_DENOMINATOR
    return round(output_power, 4)

def calculate_single_pump_efficiency(pump_output_power, pump_input_power):
    """计算单个泵的效率"""
    if pump_input_power <= POWER_EPS or pump_output_power <= POWER_EPS:
        return 0.0
    efficiency = pump_output_power / pump_input_power
    return round(efficiency, 4)

def calculate_total_efficiency(total_output_power, total_input_power):
    """计算总管效率"""
    if total_input_power <= POWER_EPS or total_output_power <= POWER_EPS:
        return 0.0
    efficiency = (total_output_power / total_input_power)
    return round(efficiency, 2)

def calculate_kwh_per_kt(total_input_power, total_flow):
    """计算千吨水电耗"""
    if total_flow <= FLOW_EPS or total_input_power <= POWER_EPS:
        return 0.0
    return round(total_input_power / (total_flow / 1000), 2)

def generate_pump_combinations(pump_group):
    """
    根据泵组运行状态生成频率组合
    pump_group: 6位字符串，如'110000'表示泵1、2开启，其他关闭
    """
    active_pumps = []
    for i, status in enumerate(pump_group[:5]):
        if status == '1':
            active_pumps.append(i)
    
    if not active_pumps:
        return [], []
    
    pump_candidates = []
    for pump_idx in active_pumps:
        pump_candidates.append(VALID_FREQUENCIES)
    
    all_combinations = list(product(*pump_candidates))
    return active_pumps, all_combinations

def save_failed_combinations(target_pressure, water_level, target_flow, pump_group, active_pumps, all_combinations):
    """保存未找到最优解工况的所有组合到txt文件"""
    os.makedirs("failed_cases", exist_ok=True)
    filename = f"failed_cases/press_{target_pressure}_flow_{target_flow}_level_{water_level}_group_{pump_group}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"=== 工况参数 ===\n")
        f.write(f"目标压力(MPa): {target_pressure}\n")
        f.write(f"目标流量(m³/h): {target_flow}\n")
        f.write(f"吸水井液位(m): {water_level}\n")
        f.write(f"泵组状态: {pump_group}\n")
        f.write(f"运行的泵: {[i+1 for i in active_pumps]}\n")
        f.write(f"频率候选值: {VALID_FREQUENCIES}\n")
        f.write(f"总组合数: {len(all_combinations)}\n\n")
        
        f.write("=== 约束条件 ===\n")
        f.write(f"流量偏差允许范围: 0-50 m³/h\n")
        f.write(f"单泵效率范围: {MIN_PUMP_EFFICIENCY}-{MAX_PUMP_EFFICIENCY}\n")
        f.write(f"总管效率范围: {MIN_TOTAL_EFFICIENCY}-{MAX_TOTAL_EFFICIENCY}\n")
        f.write(f"千吨水电耗范围: {MIN_KWH_PER_KT}-{MAX_KWH_PER_KT} kWh/kt\n\n")
        
        f.write("=== 所有组合详情 ===\n")
        
        for idx, freq_combination in enumerate(all_combinations, 1):
            full_frequencies = [0.0] * 5
            for i, pump_idx in enumerate(active_pumps):
                full_frequencies[pump_idx] = freq_combination[i]
            
            total_flow = 0.0
            total_input_power = 0.0
            total_output_power = 0.0
            pump_details = []
            valid = True
            
            for i, freq in enumerate(full_frequencies):
                if freq < FLOW_EPS:
                    continue
                
                flow = calculate_pump_flow_3d(i, freq, target_pressure)
                rated_flow = PUMP_RATED_FLOWS[i]
                
                input_power = calculate_pump_input_power(i, freq)
                output_power = calculate_single_pump_output_power(flow, target_pressure, water_level)
                efficiency = calculate_single_pump_efficiency(output_power, input_power)
                
                pump_details.append({
                    '泵编号': i + 1,
                    '频率': freq,
                    '流量': flow,
                    '额定流量': rated_flow,
                    '输入功率': input_power,
                    '输出功率': output_power,
                    '效率': efficiency,
                    '是否超额定': flow > rated_flow,
                    '效率是否合理': (MIN_PUMP_EFFICIENCY <= efficiency <= MAX_PUMP_EFFICIENCY) or efficiency ==0
                })
                
                total_flow += flow
                total_input_power += input_power
                total_output_power += output_power
            
            efficiency = calculate_total_efficiency(total_output_power, total_input_power)
            kwh_per_kt = calculate_kwh_per_kt(total_input_power, total_flow)
            
            f.write(f"\n组合 {idx}/{len(all_combinations)}:\n")
            f.write(f"频率组合: {[f'{f:.1f}' for f in full_frequencies]}\n")
            
            for detail in pump_details:
                f.write(f"泵{detail['泵编号']}:\n")
                f.write(f"  频率: {detail['频率']:.1f} Hz\n")
                f.write(f"  流量: {detail['流量']:.2f} m³/h (额定: {detail['额定流量']} m³/h)")
                f.write(f" {'[超额定]' if detail['是否超额定'] else ''}\n")
                f.write(f"  输入功率: {detail['输入功率']:.2f} kW\n")
                f.write(f"  输出功率: {detail['输出功率']:.2f} kW\n")
                f.write(f"  效率: {detail['效率']:.4f}")
                f.write(f" {'[异常]' if not detail['效率是否合理'] else ''}\n")
            
            f.write(f"总流量: {total_flow:.2f} m³/h (目标: {target_flow} m³/h, 偏差: {total_flow-target_flow:.2f} m³/h)")
            f.write(f"总输入功率: {total_input_power:.2f} kW\n")
            f.write(f"总输出功率: {total_output_power:.2f} kW\n")
            f.write(f"总管效率: {efficiency:.4f}")
            f.write(f" {'[异常]' if not ((MIN_PUMP_EFFICIENCY <= efficiency <= MAX_PUMP_EFFICIENCY) or efficiency==0.0) else ''}\n")
            f.write(f"千吨水电耗: {kwh_per_kt:.2f} kWh/kt")
            f.write(f" {'[异常]' if not ((MIN_KWH_PER_KT <= kwh_per_kt <= MAX_KWH_PER_KT) or kwh_per_kt==0.0) else ''}\n")
            
            f.write("约束检查: ")
            checks = []
            checks.append("流量✔" if target_flow <= total_flow <= target_flow + 50 else "流量✖")
            checks.append("效率✔" if ((MIN_TOTAL_EFFICIENCY <= efficiency <= MAX_TOTAL_EFFICIENCY) or efficiency==0.0) else "效率✖")
            checks.append("电耗✔" if ((MIN_KWH_PER_KT <= kwh_per_kt <= MAX_KWH_PER_KT) or kwh_per_kt==0.0) else "电耗✖")
            f.write(", ".join(checks) + "\n")
            f.write("-" * 50 + "\n")
        
        f.write(f"\n=== 结束 ===\n共 {len(all_combinations)} 个组合\n")

def find_optimal_frequencies(target_pressure, water_level, target_flow, pump_group):
    """
    暴力搜索最优频率组合
    """
    active_pumps, frequency_combinations = generate_pump_combinations(pump_group)
    
    if not frequency_combinations:
        return None, "无运行的泵"
    
    best_efficiency = -1
    best_solution = None
    valid_solutions = []
    
    for freq_combination in frequency_combinations:
        full_frequencies = [0.0] * 5
        for i, pump_idx in enumerate(active_pumps):
            full_frequencies[pump_idx] = freq_combination[i]
        
        total_flow = 0.0
        total_input_power = 0.0
        total_output_power = 0.0
        pump_details = []
        valid_solution = True
        
        for i, freq in enumerate(full_frequencies):
            if freq < FLOW_EPS:
                continue
            
            flow = calculate_pump_flow_3d(i, freq, target_pressure)
            rated_flow = PUMP_RATED_FLOWS[i]
            
            if flow > rated_flow:
                valid_solution = False
                break
            
            input_power = calculate_pump_input_power(i, freq)
            output_power = calculate_single_pump_output_power(flow, target_pressure, water_level)
            efficiency = calculate_single_pump_efficiency(output_power, input_power)
            
            if not ((MIN_PUMP_EFFICIENCY <= efficiency <= MAX_PUMP_EFFICIENCY) or efficiency==0.0):
                valid_solution = False
                break
            
            pump_details.append({
                '泵编号': i + 1,
                '频率': freq,
                '流量': flow,
                '输入功率': input_power,
                '输出功率': output_power,
                '效率': efficiency
            })
            
            total_flow += flow
            total_input_power += input_power
            total_output_power += output_power
        
        if not valid_solution:
            continue
        

        if not (target_flow <= total_flow <= target_flow + 50):
            continue
        
        
        efficiency = calculate_total_efficiency(total_output_power, total_input_power)
        if not (((MIN_TOTAL_EFFICIENCY <= efficiency <= MAX_TOTAL_EFFICIENCY))or efficiency==0.0):
            continue
        
        kwh_per_kt = calculate_kwh_per_kt(total_input_power, total_flow)
        if not ((MIN_KWH_PER_KT <= kwh_per_kt <= MAX_KWH_PER_KT)or kwh_per_kt==0.0):
            continue
        
        solution = {
            'frequencies': full_frequencies,
            'efficiency': efficiency,
            'kwh_per_kt': kwh_per_kt,
            'total_flow': total_flow,
            'total_input_power': total_input_power,
            'total_output_power': total_output_power,
            'pump_details': pump_details
        }
        
        valid_solutions.append(solution)
        
        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_solution = solution
    
    
    save_failed_combinations(target_pressure, water_level, target_flow, 
                            pump_group, active_pumps, frequency_combinations)
    
    return best_solution, f"找到{len(valid_solutions)}个有效解"

def optimize_pump_frequencies(input_file, output_file):
    """
    主优化函数
    """
    print("正在读取Excel文件...")
    df = pd.read_excel(input_file)
    
    required_columns = [
        'file2_Header_Press', 'file2_Flow_Lower', 'file2_Flow_Upper', 
        'file2_Liquid_Level', 'file2_Pump_Group'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要的列: {missing_columns}")
    
    print(f"找到 {len(df)} 行数据需要优化")
    df['file2_Pump_Group_Formatted'] = df['file2_Pump_Group'].apply(format_pump_group)
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="优化进度"):
        target_pressure = row['file2_Header_Press']
        flow_lower = row['file2_Flow_Lower']
        flow_upper = row['file2_Flow_Upper']
        water_level = row['file2_Liquid_Level']
        pump_group = row['file2_Pump_Group_Formatted']
        target_flow = flow_lower
        
        best_solution, message = find_optimal_frequencies(
            target_pressure, water_level, target_flow, pump_group
        )
        
        result_row = {
            'file1_Header_Press': target_pressure,
            'file1_Flow_Lower': flow_lower,
            'file1_Flow_Upper': flow_upper,
            'file1_Liquid_Level': water_level,
            'file1_Pump_Group': pump_group,
            'optimization_status': message
        }
        
        if best_solution is not None:
            for i in range(5):
                result_row[f'file1_P{i+1}_Freq'] = best_solution['frequencies'][i]
            
            pump_eff_dict = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}
            pump_flow_dict = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}
            
            for detail in best_solution['pump_details']:
                pump_id = detail['泵编号']
                pump_flow_dict[pump_id] = detail['流量']
                pump_eff_dict[pump_id] = detail['效率']
            
            for i in range(1, 6):
                result_row[f'file1_P{i}_Flow'] = pump_flow_dict[i]
                result_row[f'file1_P{i}_Eff'] = pump_eff_dict[i]
            
            result_row['file1_Total_Flow'] = best_solution['total_flow']
            result_row['file1_Eff'] = best_solution['efficiency']
            result_row['file1_PCTW'] = best_solution['kwh_per_kt']
            result_row['file1_P6_Freq'] = 0.0
            result_row['file1_P6_Flow'] = 0.0
            result_row['file1_P6_Eff'] = 0.0
        else:
            for i in range(1, 7):
                result_row[f'file1_P{i}_Freq'] = 0.0
                result_row[f'file1_P{i}_Flow'] = 0.0
                result_row[f'file1_P{i}_Eff'] = 0.0
            
            result_row['file1_Total_Flow'] = 0.0
            result_row['file1_Flow_Deviation'] = 0.0
            result_row['file1_Eff'] = 0.0
            result_row['file1_PCTW'] = 0.0
        
        results.append(result_row)
    
    results_df = pd.DataFrame(results)
    
    output_columns = [
        'file1_Header_Press', 'file1_Flow_Lower', 'file1_Flow_Upper', 'file1_Liquid_Level',
        'file1_Pump_Group', 
        'file1_P1_Freq', 'file1_P2_Freq', 'file1_P3_Freq', 'file1_P4_Freq', 'file1_P5_Freq', 'file1_P6_Freq',
        'file1_P1_Flow', 'file1_P2_Flow', 'file1_P3_Flow', 'file1_P4_Flow', 'file1_P5_Flow', 'file1_P6_Flow',
        'file1_Total_Flow', 'file1_Flow_Deviation',
        'file1_P1_Eff', 'file1_P2_Eff', 'file1_P3_Eff', 'file1_P4_Eff', 'file1_P5_Eff', 'file1_P6_Eff',
        'file1_Eff', 'file1_PCTW', 'optimization_status'
    ]
    
    for col in output_columns:
        if col not in results_df.columns:
            results_df[col] = 0.0
    
    results_df = results_df[output_columns]
    results_df.to_excel(output_file, index=False)
    
    if os.path.exists("failed_cases"):
        failed_files = os.listdir("failed_cases")
        print(f"\n共保存了 {len(failed_files)} 个失败工况的详细组合信息")
        print(f"详细组合信息保存在: {os.path.abspath('failed_cases')}")
    
    return results_df

if __name__ == "__main__":
    input_excel = "历史策略独有数据_20251225_193359.xlsx"  # 请修改为实际文件路径
    output_excel = "优化结果_20251225_193359.xlsx"     # 输出文件路径
    
    try:
        results = optimize_pump_frequencies(input_excel, output_excel)
        print("\n优化完成！前5行结果预览：")
        print(results[['file1_Pump_Group', 'file1_Eff', 'optimization_status']].head())
    except Exception as e:
        print(f"优化过程中出错: {e}")
        import traceback
        traceback.print_exc()
