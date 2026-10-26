import numpy as np
import pandas as pd
import time
import multiprocessing
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product, combinations

# ====== 1. 预计算常量 ======
DENSITY_WATER = 1000  # kg/m³
GRAVITY = 9.81        # m/s²
PRESSURE_TO_HEAD = 102  # 1MPa = 102m扬程
EFFICIENCY_DENOMINATOR = 3600000  # 输出功率计算分母
FLOW_EPS = 0  # 流量极小值阈值
POWER_EPS = 0  # 功率极小值阈值

# 泵额定功率 (kW)
PUMP_RATED_POWER = np.array([315, 315, 315, 315, 710])  # 转为数组，支持向量化
PUMP_RATED_FLOWS = [1460, 1460, 1460, 1460, 2910]  # 各泵额定流量(m³/h)

# ====== 各泵运行压力区间（MPa）======
# 格式：{泵索引: (最小压力, 最大压力)}
PUMP_PRESSURE_RANGE = {
    0: (0.30, 0.60),  # 泵1：
    1: (0.30, 0.60),  # 泵2：
    2: (0.30, 0.60),  # 泵3：
    3: (0.30, 0.60),  # 泵4：
    4: (0.30, 0.60)   # 泵5：
}

POWER_FACTORS = [0.89, 0.89, 0.86, 0.89, 0.83, 0.83]

# 约束条件
MAX_FLOW_DEVIATION_ABS = 0.0  # 流量偏差绝对值
MIN_PUMP_EFFICIENCY = 0.4  # 单泵效率≥0.4
MAX_PUMP_EFFICIENCY = 0.9  # 单泵效率≤0.9
MIN_KWH_PER_KT = 100.0       # 千吨水电耗下限
MAX_KWH_PER_KT = 200.0       # 千吨水电耗上限

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

# 遍历参数范围
TARGET_FLOWS = sorted(np.arange(1000, 8801, 50))          # 总管流量：1000~8800 m³
TARGET_PRESSURES = sorted(np.arange(0.35, 0.601, 0.01))    # 总管压力：0.35~0.6 MPa
WATER_LEVELS = sorted(np.arange(3.5, 6.6, 0.5))            # 吸水井液位：3.5~6.5 m
VALID_FREQUENCIES = list(np.arange(40, 50.5, 1))

# ====== 2. 核心计算函数 ======
def check_pump_pressure_valid(pump_idx, pressure):
    """检查单个泵在指定压力下是否可运行"""
    if pump_idx not in PUMP_PRESSURE_RANGE:
        return False  # 未知泵默认不可运行
    min_p, max_p = PUMP_PRESSURE_RANGE[pump_idx]
    return min_p <= pressure <= max_p

def check_combination_pressure_valid(pump_status, pressure):
    """检查当前泵组合中所有运行泵的压力是否都在各自的运行区间内"""
    invalid_pumps = []
    active_pumps = [i for i, s in enumerate(pump_status) if s == 1]
    
    for pump_idx in active_pumps:
        if not check_pump_pressure_valid(pump_idx, pressure):
            invalid_pumps.append(pump_idx + 1)  # 转为泵编号（1-5）
    
    return len(invalid_pumps) == 0, invalid_pumps

def calculate_pump_flow_3d(pump_idx, frequency, actual_pressure):
    """基于三维多项式函数计算单个泵的出口流量"""
    if frequency <= FLOW_EPS:
        return 0.0

    coeffs = PUMP_3D_FLOW_FUNC[pump_idx]['coeffs']
    f, p = frequency, actual_pressure

    # 预计算幂次减少重复计算
    f2, f3 = f**2, f**3
    p2, p3 = p**2, p**3

    flow = (coeffs[0] + coeffs[1]*f + coeffs[2]*p + coeffs[3]*f2 + coeffs[4]*p2 +
            coeffs[5]*f*p + coeffs[6]*f3 + coeffs[7]*p3 + coeffs[8]*f2*p + coeffs[9]*f*p2)
    return max(flow, 0.0)

def calculate_pump_input_power(pump_idx, frequency):
    """计算单个泵的输入功率"""
    if frequency <= POWER_EPS:
        return 0.0
    # 获取额定功率和功率因子
    rated_power = PUMP_RATED_POWER[pump_idx]
    power_factor = POWER_FACTORS[pump_idx]

    # 计算基础功率
    base_power = rated_power * (frequency / 50) ** 3

    # 应用功率因子
    return base_power * power_factor

def calculate_single_pump_output_power(pump_flow, total_pressure, water_level):
    """计算单个泵的输出功率"""
    if pump_flow <= FLOW_EPS:
        return 0.0

    H = total_pressure * PRESSURE_TO_HEAD
    effective_head = H - (water_level - 2.35)
    if effective_head < 0:
        return 0.0

    output_power = (DENSITY_WATER * GRAVITY * pump_flow *
                    effective_head) / EFFICIENCY_DENOMINATOR
    return round(output_power, 4)

def calculate_single_pump_efficiency(pump_output_power, pump_input_power):
    """计算单个泵的效率"""
    efficiency = (pump_output_power / pump_input_power)
    return round(efficiency, 2)

def calculate_total_efficiency(total_output_power, total_input_power):
    """计算总管效率"""
    if total_input_power < POWER_EPS or total_output_power < POWER_EPS:
        return 0.0
    efficiency = (total_output_power / total_input_power)
    return round(efficiency, 2)

def calculate_kwh_per_kt(total_input_power, total_flow):
    """计算千吨水电耗"""
    if total_flow < FLOW_EPS or total_input_power < POWER_EPS:
        return 0.0
    return round(total_input_power / (total_flow / 1000), 2)

def format_seconds(seconds):
    """将秒数格式化为 时:分:秒 字符串"""
    return str(timedelta(seconds=round(seconds)))

def generate_running_combination_str(pump_status):
    """生成运行组合字符串"""
    return ''.join(str(s) for s in pump_status) + '0'  # 泵6始终关闭

def generate_pump_combinations():
    """生成所有可能的泵运行组合"""
    return [[(i >> j) & 1 for j in range(5)] for i in range(1, 2**5)]

def is_possible_combination(pump_status, target_flow, target_pressure):
    """判断泵组合是否可能满足流量需求和压力约束"""
    active_pumps = [i for i, s in enumerate(pump_status) if s == 1]
    if not active_pumps:
        return False  # 无泵运行

    # 检查压力约束：所有运行泵都必须在各自的压力区间内
    for pump_idx in active_pumps:
        if not check_pump_pressure_valid(pump_idx, target_pressure):
            return False  # 有泵超出压力运行区间

    # 检查流量约束：流量必须在最小和最大可能流量之间
    total_min_flow = sum(
        PUMP_RATED_FLOWS[i] for i in active_pumps) * 0
    
    total_max_flow = sum(PUMP_RATED_FLOWS[i]
                         for i in active_pumps)

    return total_min_flow <= target_flow <= total_max_flow

# ====== 3. 暴力搜索优化器 ======
class BruteForcePumpOptimizer:
    def __init__(self, min_total_efficiency, max_total_efficiency):
        # 从命令行传入的总管效率上下限
        self.MIN_TOTAL_EFFICIENCY = min_total_efficiency
        self.MAX_TOTAL_EFFICIENCY = max_total_efficiency

    def generate_pump_candidates(self, pump_idx, target_pressure, water_level):
        """为单个泵生成频率候选点（先检查压力有效性）"""
        # 先检查该泵在当前压力下是否可运行
        if not check_pump_pressure_valid(pump_idx, target_pressure):
            return []  # 不可运行则返回空列表
        
        candidates = []
        for freq in VALID_FREQUENCIES:
            flow = calculate_pump_flow_3d(pump_idx, freq, target_pressure)
            input_power = calculate_pump_input_power(pump_idx, freq)
            output_power = calculate_single_pump_output_power(
                flow, target_pressure, water_level)
            efficiency = calculate_single_pump_efficiency(
                output_power, input_power)

            candidates.append({
                '泵编号': pump_idx + 1,
                '频率(Hz)': freq,
                '出口流量': flow,
                '输入功率(kW)': input_power,
                '输出功率(kW)': output_power,
                '效率': efficiency
            })
        
        return candidates

    def find_optimal_combination(self, pump_status, target_flow, target_pressure, water_level, combo_idx):
        """寻找最优泵组合"""
        start_time = time.time()
        
        # 打印当前工况信息
        print(f"\n=== 工况 {combo_idx} ===")
        print(f"目标流量: {target_flow} m³/h | 目标压力: {target_pressure} MPa | 液位: {water_level} m")
        print(f"泵状态: {pump_status}")
        
        # 第一步：使用整合后的函数检查泵组合有效性
        if not is_possible_combination(pump_status, target_flow, target_pressure):
            print(f"⚠️ 组合校验失败：泵组合在压力{target_pressure}MPa下不可运行或流量{target_flow}m³/h无法满足，跳过计算")
            return {
                'combo_idx': combo_idx,
                'solution': None,
                'elapsed': time.time() - start_time,
                'params': (target_pressure, target_flow, water_level, pump_status),
                'total_combinations': 0,
                'skip_reason': f"压力或流量约束不满足"
            }
        
        active_pumps = [i for i, s in enumerate(pump_status) if s == 1]
        if not active_pumps:
            print("⚠️ 无泵运行，跳过计算")
            return {
                'combo_idx': combo_idx,
                'solution': None,
                'elapsed': time.time() - start_time,
                'params': (target_pressure, target_flow, water_level, pump_status),
                'total_combinations': 0,
                'skip_reason': "无泵运行"
            }
        
        print(f"active_pumps is {active_pumps}")

        # 步骤1：生成候选点（已包含单泵压力检查）
        all_candidates = []
        invalid_pump_count = 0
        for pump_idx in active_pumps:
            candidates = self.generate_pump_candidates(pump_idx, target_pressure, water_level)
            if not candidates:
                print(f"⚠️ 泵{pump_idx+1}在当前压力下无有效候选点，工况无解")
                return {
                    'combo_idx': combo_idx,
                    'solution': None,
                    'elapsed': time.time() - start_time,
                    'params': (target_pressure, target_flow, water_level, pump_status),
                    'total_combinations': 0,
                    'skip_reason': f"泵{pump_idx+1}无有效候选点（压力超限）"
                }
            all_candidates.append(candidates)
            print(f"泵{pump_idx+1}候选频率数: {len(candidates)}")

        # 计算总组合数
        total_combinations = 1
        for candidates in all_candidates:
            total_combinations *= len(candidates)
        print(f"理论总组合数: {total_combinations:,}")
        
        # 步骤2：搜索最优组合
        best_solution = None
        best_total_power = float('inf')
        evaluated_combinations = 0
        last_print_time = time.time()
        
        for candidate_combination in product(*all_candidates):
            evaluated_combinations += 1
            
            # 每5秒打印一次进度（避免刷屏）
            current_time = time.time()
            if current_time - last_print_time > 5:
                progress = evaluated_combinations / total_combinations * 100
                print(f"进度: {evaluated_combinations:,}/{total_combinations:,} ({progress:.1f}%)")
                last_print_time = current_time
            
            # 原有计算逻辑
            total_flow = 0.0
            total_input_power = 0.0
            total_output_power = 0.0
            valid = True
            pump_details = []
            
            for candidate in candidate_combination:
                if candidate['效率'] > MAX_PUMP_EFFICIENCY or candidate['效率'] < MIN_PUMP_EFFICIENCY:
                    valid = False
                    break
                pump_idx = candidate['泵编号'] - 1
                rated_flow = PUMP_RATED_FLOWS[pump_idx]

                if candidate['出口流量'] > rated_flow:
                    valid = False
                    break

                total_flow += candidate['出口流量']
                total_input_power += candidate['输入功率(kW)']
                total_output_power += candidate['输出功率(kW)']
                pump_details.append(candidate)
            
            if not valid:
                continue
            
            flow_deviation_abs = abs(total_flow - target_flow)
            
          
            if not (target_flow <= total_flow <= (target_flow + 50)):
                continue  # 流量偏差过大，跳过
            
            
            total_efficiency = calculate_total_efficiency(total_output_power, total_input_power)
            kwh_per_kt = calculate_kwh_per_kt(total_input_power, total_flow)
            
            if not (self.MIN_TOTAL_EFFICIENCY <= total_efficiency <= self.MAX_TOTAL_EFFICIENCY):
                continue
            if not (MIN_KWH_PER_KT <= kwh_per_kt <= MAX_KWH_PER_KT):
                continue
            
            if total_input_power < best_total_power:
                best_total_power = total_input_power
                best_solution = {
                    '总流量': round(total_flow, 2),
                    '总输入功率(kW)': round(total_input_power, 2),
                    '总输出功率(kW)': round(total_output_power, 4),
                    '总管效率': total_efficiency,
                    '千吨水电耗(kWh/kt)': kwh_per_kt,
                    '单泵详情': pump_details,
                    '泵状态': pump_status
                }
        
        # 最终结果输出
        elapsed = time.time() - start_time
        pruned = total_combinations - evaluated_combinations
        print("\n=== 计算结果 ===")
        print(f"实际评估组合数: {evaluated_combinations:,}")
        
        print(f"计算耗时: {elapsed:.2f}秒")
        if best_solution:
            print(f"最优解总功率: {best_solution['总输入功率(kW)']} kW")
        else:
            print("⚠️ 未找到可行解")
        
        return {
            'combo_idx': combo_idx,
            'solution': best_solution,
            'elapsed': elapsed,
            'params': (target_pressure, target_flow, water_level, pump_status),
            'total_combinations': total_combinations,
            'evaluated_combinations': evaluated_combinations,
            'pruned_combinations': pruned
        }

# ====== 4. 结果格式化 ======
def format_result(solution, pump_status, target_flow, target_pressure, water_level):
    """格式化输出结果（仅处理有有效解的情况）"""
    if not solution:
        return None  # 无有效解时返回None，后续过滤
    
    pump_freqs = [0.0] * 6
    pump_effs = [0.0] * 6
    pump_out_flows = [0.0] * 6

    # 计算所有泵出口流量的总和
    total_pump_flow = 0.0

    for pump in solution['单泵详情']:
        pump_id = pump['泵编号'] - 1
        if 0 <= pump_id < 5:
            pump_freqs[pump_id] = pump['频率(Hz)']
            pump_effs[pump_id] = pump['效率']
            pump_out_flows[pump_id] = pump['出口流量']
            total_pump_flow += pump['出口流量']

    return {
        'Header_Press': round(target_pressure, 2),  # 总管压力
        'Flow_Lower': round(target_flow, 2),  # 流量下限
        'Flow_Upper': round(target_flow + 50, 2),  # 流量上限
        'Liquid_Level': round(water_level, 1),  # 液位
        'Pump_Group': generate_running_combination_str(solution['泵状态']),  # 泵组
        'P1_Freq': pump_freqs[0],  # 1号泵给定频率
        'P2_Freq': pump_freqs[1],  # 2号泵给定频率
        'P3_Freq': pump_freqs[2],  # 3号泵给定频率
        'P4_Freq': pump_freqs[3],  # 4号泵给定频率
        'P5_Freq': pump_freqs[4],  # 5号泵给定频率
        'P6_Freq': pump_freqs[5],  # 6号泵给定频率
        'P1_Flow': pump_out_flows[0],  # 1号泵出口流量
        'P2_Flow': pump_out_flows[1],  # 2号泵出口流量
        'P3_Flow': pump_out_flows[2],  # 3号泵出口流量
        'P4_Flow': pump_out_flows[3],  # 4号泵出口流量
        'P5_Flow': pump_out_flows[4],  # 5号泵出口流量
        'P6_Flow': pump_out_flows[5],  # 6号泵出口流量
        'Total_Flow': round(total_pump_flow, 2),  # 所有泵出口流量总和
        'Flow_Deviation': round(abs(total_pump_flow - target_flow), 2),
        'P1_Eff': pump_effs[0],  # 1号泵运行效率
        'P2_Eff': pump_effs[1],  # 2号泵运行效率
        'P3_Eff': pump_effs[2],  # 3号泵运行效率
        'P4_Eff': pump_effs[3],  # 4号泵运行效率
        'P5_Eff': pump_effs[4],  # 5号泵运行效率
        'P6_Eff': pump_effs[5],  # 6号泵运行效率
        'Eff': solution['总管效率'],  # 整体效率
        'PCTW': solution['千吨水电耗(kWh/kt)']  # 千吨水电耗
    }

# ====== 5. 主函数 ======
def main():
    """主函数：全组合优化计算"""
    # 第一步：解析命令行参数
    parser = argparse.ArgumentParser(description='泵组优化计算程序')
    
    # 修改：将required=True改为默认值0.4和0.9[1,2](@ref)
    parser.add_argument('--min-eff',
                        type=float,
                        default=0.4,  # 设置默认值0.4
                        help='总管效率下限 (默认: 0.4)')
    parser.add_argument('--max-eff',
                        type=float,
                        default=0.9,  # 设置默认值0.9
                        help='总管效率上限 (默认: 0.9)')

    # 解析参数
    args = parser.parse_args()
    MIN_TOTAL_EFFICIENCY = args.min_eff
    MAX_TOTAL_EFFICIENCY = args.max_eff

    # 验证参数合理性
    if MIN_TOTAL_EFFICIENCY < 0.0 or MAX_TOTAL_EFFICIENCY > 1.0:
        print("❌ 错误：总管效率上下限必须在0.0~1.0之间！")
        return
    if MIN_TOTAL_EFFICIENCY >= MAX_TOTAL_EFFICIENCY:
        print("❌ 错误：总管效率下限必须小于上限！")
        return

    # 遍历参数范围
    pump_combinations = generate_pump_combinations()

    print("="*80)
    print("泵组优化计算程序")
    print("="*80)
    print(f"当前配置的总管效率约束：{MIN_TOTAL_EFFICIENCY} ~ {MAX_TOTAL_EFFICIENCY}")
    print(f"各泵运行压力区间（MPa）：")
    for pump_idx, (min_p, max_p) in PUMP_PRESSURE_RANGE.items():
        print(f"  泵{pump_idx+1}：{min_p} ~ {max_p}")
    print(f"遍历参数范围：")
    print(f"  总管压力：{TARGET_PRESSURES[0]} ~ {TARGET_PRESSURES[-1]} MPa")
    print(f"  总管流量：{TARGET_FLOWS[0]} ~ {TARGET_FLOWS[-1]} m³")
    print(f"  吸水井液位：{WATER_LEVELS[0]} ~ {WATER_LEVELS[-1]} m")
    print(f"  泵运行组合：31种 (2^5-1，泵6始终关闭)")

    # 生成所有任务参数
    all_task_args = []
    combo_seq = 1
    for target_pressure in TARGET_PRESSURES:
        for target_flow in TARGET_FLOWS:
            for water_level in WATER_LEVELS:
                for pump_status in pump_combinations:
                    # 使用整合后的函数检查泵组合是否可能满足当前流量和压力
                    if is_possible_combination(pump_status, target_flow, target_pressure):
                        all_task_args.append(
                            (pump_status, target_flow, target_pressure, water_level, combo_seq))
                        combo_seq += 1

    total_combs = len(all_task_args)
    print(f"  总记录数：{total_combs} 条")
    print(f"  线程数：{min(32, multiprocessing.cpu_count() * 2)}")
    print("="*80)

    # 新增统计变量
    valid_solutions = 0
    invalid_solutions = 0
    pressure_skip_count = 0  

    # 正式运行
    all_results = []
    start_total_time = time.time()

    print("\n🚀 开始优化计算...")
    print("="*80)

    # 实例化优化器时传入命令行读取的效率上下限
    optimizer = BruteForcePumpOptimizer(
        min_total_efficiency=MIN_TOTAL_EFFICIENCY,
        max_total_efficiency=MAX_TOTAL_EFFICIENCY
    )

    # 使用tqdm包装任务列表
    with ThreadPoolExecutor(max_workers=min(32, multiprocessing.cpu_count() * 2)) as executor:
        futures = {executor.submit(
            optimizer.find_optimal_combination, *args): args for args in all_task_args}

        # 使用tqdm显示进度
        with tqdm(total=total_combs, desc="优化进度", unit="组合") as pbar:
            for future in as_completed(futures):
                result = future.result()
                combo_idx = result['combo_idx']

                # 更新统计计数
                if result.get('skip_reason') and "压力超限" in result['skip_reason']:
                    pressure_skip_count += 1
                    invalid_solutions += 1
                elif result.get('skip_reason'):
                    invalid_solutions += 1
                elif result['solution']:
                    valid_solutions += 1
                    # 仅当有有效解时才格式化并添加到结果列表
                    formatted_result = format_result(
                        result['solution'],
                        result['params'][3],  # pump_status
                        result['params'][1],  # target_flow
                        result['params'][0],  # target_pressure
                        result['params'][2]   # water_level
                    )
                    if formatted_result:  # 双重确认：确保结果非空
                        all_results.append(formatted_result)
                else:
                    invalid_solutions += 1

                # 更新进度条
                pbar.update(1)

                # 实时显示统计信息
                pbar.set_postfix({
                    '有效解': valid_solutions,
                    '无效解': invalid_solutions,
                    '压力超限跳过': pressure_skip_count,
                    '进度': f"{len(all_results)}/{total_combs}"  # 显示已生成的有效解数量
                })

    # 保存结果
    end_total_time = time.time()
    actual_total_sec = end_total_time - start_total_time

    # 直接使用有效结果列表生成DataFrame（无空值）
    df_results = pd.DataFrame(all_results)
    if not df_results.empty:  # 确保有数据时才排序和保存
        # 更新列顺序，包含新增的泵出口流量字段[9,10](@ref)
        column_order = [
            'Header_Press', 'Flow_Lower', 'Flow_Upper', 'Liquid_Level', 'Pump_Group',
            'P1_Freq', 'P2_Freq', 'P3_Freq', 'P4_Freq', 'P5_Freq', 'P6_Freq',
            'P1_Flow', 'P2_Flow', 'P3_Flow', 'P4_Flow', 'P5_Flow', 'P6_Flow', 'Total_Flow','Flow_Deviation',
            'P1_Eff', 'P2_Eff', 'P3_Eff', 'P4_Eff', 'P5_Eff', 'P6_Eff',
            'Eff', 'PCTW'
        ]
        df_results = df_results[column_order]

        df_results = df_results.sort_values(
            by=['Header_Press', 'Flow_Lower', 'Liquid_Level', 'Pump_Group'],
            ascending=[True, True, True, True]
        ).reset_index(drop=True)

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 文件名中包含效率上下限和压力约束标识
    output_file = f"计算最优策略_效率{MIN_TOTAL_EFFICIENCY}-{MAX_TOTAL_EFFICIENCY}_{time_str}.csv"
    # 保存时指定UTF-8-SIG编码，避免中文乱码，且仅保存有效数据[9,11](@ref)
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')

    # 计算有效解比例
    valid_ratio = (valid_solutions / total_combs) * 100 if total_combs > 0 else 0
    pressure_skip_ratio = (pressure_skip_count / total_combs) * 100 if total_combs > 0 else 0

    print("\n" + "="*80)
    print("✅ 优化计算完成！")
    print("="*80)
    print(f"📊 最终统计：")
    print(f"   总参数组合数：{total_combs} 条")
    print(f"   有效解数量：{valid_solutions} 条 ({valid_ratio:.1f}%)")
    print(f"   无效解数量：{invalid_solutions} 条")
    print(f"   压力超限跳过数：{pressure_skip_count} 条 ({pressure_skip_ratio:.1f}%)")
    print(f"   实际输出记录数：{len(all_results)} 条（仅包含有效解）")
    print(f"   实际总时长：{format_seconds(actual_total_sec)}")
    print(f"   平均单组合耗时：{actual_total_sec/total_combs:.2f}秒")
    print(f"📁 结果已保存至：{output_file}")
    print("="*80)

if __name__ == "__main__":
    # 安装进度条库（如果尚未安装）
    try:
        from tqdm import tqdm
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        from tqdm import tqdm

    main()