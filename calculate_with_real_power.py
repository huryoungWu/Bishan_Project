import os
import numpy as np
import pandas as pd
import time
import multiprocessing
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product, combinations

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
MIN_PUMP_EFFICIENCY = 0.0  # 单泵效率≥0.4
MAX_PUMP_EFFICIENCY = 5.0  # 单泵效率≤0.9
MIN_KWH_PER_KT = 0.0       # 千吨水电耗下限
MAX_KWH_PER_KT = 500.0       # 千吨水电耗上限

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

# # 遍历参数范围
# TARGET_FLOWS = sorted(np.arange(1000, 8801, 50))          # 总管流量：1000~8800 m³
# TARGET_PRESSURES = sorted(np.arange(0.35, 0.601, 0.01))    # 总管压力：0.35~0.6 MPa
# WATER_LEVELS = sorted(np.arange(3.5, 6.6, 0.5))            # 吸水井液位：3.5~6.5 m
# VALID_FREQUENCIES = list(np.arange(40, 50.5, 1))

TARGET_FLOWS = [2000]       # 总管流量：1000~8800 m³
TARGET_PRESSURES = [0.4]     # 总管压力：0.35~0.6 MPa
WATER_LEVELS = [3]          # 吸水井液位：3.5~6.5 m
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

def calculate_pump_input_power(pump_idx, frequency, pressure, flow):
    """基于三维多项式函数计算单个泵的输入功率"""
    if frequency <= POWER_EPS:
        return 0.0

    # 各泵的功率多项式系数
    PUMP_POWER_COEFFS = {
        0: [-0.000123277200227790516675, -0.000569330709645632891680, -0.009345111448190919836709, -0.003252405509154680102180,
            -0.000767086019755507174216, -0.000112591754977554113341, -0.060604180653720099436477, -0.237306446788038194473103,
            -0.015384668157856499354219, 0.008479552283737061199598, -0.000796426288524317211627, -0.000000102543554741993654,
            0.006245257559796034765864, -0.457653792409820292519385, -0.000546502144615462217246, -0.025599262187301505266923,
            -0.242087959036627387021667, 0.000015273355638346972193, -0.000625238681096900524199, 0.038255541100780290908467],
        1: [-0.000073910987873009796438, 0.000194342360893723560024, 0.046156560469413326908139, -0.003022724993287399450348,
            0.000329359765175312423080, -0.000004216810185948286426, -0.095267549799927395137722, 0.170557531448702365262093,
            0.003492882480040250710307, 0.000164709271291127473018, 0.000394402342213584587022, 0.000000161059952029482383,
            0.001924656859179380485533, 0.328922108168564597541206, 0.000417486938238311974022, 0.011977476745335346314247,
            -0.012513678777760829999299, -0.000015364674504996853050, 0.000384482097395157378268, -0.022477379502439491509858],
        2: [0.000000339005073627037617, 0.000000245163898937770251, 0.000200533390095359171897, 0.000004244716085771999381,
            -0.000000098287121980804213, 0.000000078366838029506245, -0.000037445577681154246469, 0.000202670474299833602806,
            0.000002896653003127236236, -0.000012690226522909657829, -0.000000277478765955997635, -0.000000000008077198093281,
            0.000003559808845473767547, -0.000188855785711433828925, -0.000000062828808732329802, -0.000009676396376204823361,
            -0.000117308674532895897483, -0.000000001036100520863762, 0.000000148955453978569701, 0.000003731444930475524406],
        3: [-0.000811448793955587351252, -0.001057533260067752530278, -0.093630605469058852952102, -0.011777270470190016535272,
            -0.001007055158755538202056, -0.000219645508645835413732, -0.074628129973055137846849, -0.362310028313356013995872,
            -0.026083886909123363967433, 0.017651561087665072757869, -0.000830490550621201003949, -0.000000166365264730419909,
            0.010031837927067147053783, -0.497485121771721883199291, -0.000938119880061218189976, -0.031425665148588570463328,
            -0.487350300039461503676108, 0.000026612458609022050641, -0.001120691292324151912274, 0.062018993759671930954269],
        4: [0.000000244939534548334956, -0.000000018024771011994212, 0.000214082498255918689213, 0.000003738376533516398684,
            -0.000000015835719908385727, -0.000000038001794923231547, 0.000003681904455146222049, -0.000076837781528554406477,
            -0.000001245578095826998962, -0.000002587344866005386814, 0.000000014091532023643237, 0.000000000006580016253187,
            0.000003888959468418016993, 0.000028309234436768181722, 0.000000097261328516600112, 0.000000444585147322910570,
            -0.000000951570901169838711, -0.000000001017633446068139, 0.000000068996965028421553, -0.000005851856162254283359]
    }

    coeffs = PUMP_POWER_COEFFS[pump_idx]
    f, p, q = frequency, pressure, flow

    # 预计算幂次减少重复计算
    f2, f3 = f**2, f**3
    p2, p3 = p**2, p**3
    q2, q3 = q**2, q**3

    # 计算功率
    power = (coeffs[0] + coeffs[1]*p + coeffs[2]*q + coeffs[3]*f +
             coeffs[4]*p2 + coeffs[5]*q2 + coeffs[6]*f2 +
             coeffs[7]*p*q + coeffs[8]*p*f + coeffs[9]*q*f +
             coeffs[10]*p3 + coeffs[11]*q3 + coeffs[12]*f3 +
             coeffs[13]*p2*q + coeffs[14]*p*q2 +
             coeffs[15]*p2*f + coeffs[16]*p*f2 +
             coeffs[17]*q2*f + coeffs[18]*q*f2 +
             coeffs[19]*p*q*f)

    return max(power, 0.0)

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
    if pump_output_power <= 0 or pump_input_power <= 0:
        return 0.0
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

def save_failed_combinations(target_pressure, water_level, target_flow, pump_group, active_pumps, 
                           all_candidates, total_combinations_count):
    """保存未找到最优解工况的所有组合到txt文件"""
    os.makedirs("failed_cases", exist_ok=True)
    filename = f"failed_cases/press_{target_pressure}_flow_{target_flow}_level_{water_level}_group_{''.join(map(str, pump_group))}0.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"=== 工况参数 ===\n")
        f.write(f"目标压力(MPa): {target_pressure}\n")
        f.write(f"目标流量(m³/h): {target_flow}\n")
        f.write(f"吸水井液位(m): {water_level}\n")
        f.write(f"泵组状态: {''.join(map(str, pump_group))}0\n")  # 泵6始终关闭
        f.write(f"运行的泵: {[i+1 for i in active_pumps]}\n")
        f.write(f"频率候选值: {VALID_FREQUENCIES}\n")
        f.write(f"总组合数: {total_combinations_count}\n\n")
        
        f.write("=== 约束条件 ===\n")
        f.write(f"流量偏差允许范围: 0-50 m³/h\n")
        f.write(f"单泵效率范围: {MIN_PUMP_EFFICIENCY}-{MAX_PUMP_EFFICIENCY}\n")
        f.write(f"总管效率范围: {MIN_TOTAL_EFFICIENCY}-{MAX_TOTAL_EFFICIENCY}\n")
        f.write(f"千吨水电耗范围: {MIN_KWH_PER_KT}-{MAX_KWH_PER_KT} kWh/kt\n\n")
        
        f.write("=== 所有组合详情 ===\n")
        
        # 由于组合数量可能很大，我们只记录部分组合或统计信息
        from itertools import product
        
        # 限制记录的组合数量以避免文件过大
        max_combinations_to_record = 100  # 可根据需要调整
        recorded_count = 0
        
        for idx, candidate_combination in enumerate(product(*all_candidates)):
            if recorded_count >= max_combinations_to_record:
                f.write(f"\n... 还有 {total_combinations_count - max_combinations_to_record} 个组合未显示 ...\n")
                break
                
            recorded_count += 1
            f.write(f"\n组合 {idx+1}/{total_combinations_count}:\n")
            
            total_flow = 0.0
            total_input_power = 0.0
            total_output_power = 0.0
            pump_details = []
            
            for candidate in candidate_combination:
                pump_idx = candidate['泵编号'] - 1
                rated_flow = PUMP_RATED_FLOWS[pump_idx]

                flow = candidate['出口流量']
                input_power = candidate['输入功率(kW)']
                output_power = candidate['输出功率(kW)']
                efficiency = candidate['效率']

                total_flow += flow
                total_input_power += input_power
                total_output_power += output_power
                pump_details.append(candidate)
            
            efficiency = calculate_total_efficiency(total_output_power, total_input_power)
            kwh_per_kt = calculate_kwh_per_kt(total_input_power, total_flow)
            
            # 记录频率组合
            freq_combination = [candidate['频率(Hz)'] for candidate in candidate_combination]
            f.write(f"频率组合: {[f'{f:.1f}' for f in freq_combination]}\n")
            
            for detail in pump_details:
                f.write(f"泵{detail['泵编号']}:\n")
                f.write(f"  频率: {detail['频率(Hz)']:.1f} Hz\n")
                f.write(f"  流量: {detail['出口流量']:.2f} m³/h (额定: {PUMP_RATED_FLOWS[detail['泵编号']-1]} m³/h)")
                f.write(f" {'[超额定]' if detail['出口流量'] > PUMP_RATED_FLOWS[detail['泵编号']-1] else ''}\n")
                f.write(f"  输入功率: {detail['输入功率(kW)']:.2f} kW\n")
                f.write(f"  输出功率: {detail['输出功率(kW)']:.2f} kW\n")
                f.write(f"  效率: {detail['效率']:.4f}")
                f.write(f" {'[异常]' if not ((MIN_PUMP_EFFICIENCY <= detail['效率'] <= MAX_PUMP_EFFICIENCY) or detail['效率']==0) else ''}\n")
            
            f.write(f"总流量: {total_flow:.2f} m³/h (目标: {target_flow} m³/h, 偏差: {total_flow-target_flow:.2f} m³/h)")
            f.write(f"总输入功率: {total_input_power:.2f} kW\n")
            f.write(f"总输出功率: {total_output_power:.2f} kW\n")
            f.write(f"总管效率: {efficiency:.4f}")
            f.write(f" {'[异常]' if not ((MIN_TOTAL_EFFICIENCY <= efficiency <= MAX_TOTAL_EFFICIENCY) or efficiency==0.0) else ''}\n")
            f.write(f"千吨水电耗: {kwh_per_kt:.2f} kWh/kt")
            f.write(f" {'[异常]' if not ((MIN_KWH_PER_KT <= kwh_per_kt <= MAX_KWH_PER_KT) or kwh_per_kt==0.0) else ''}\n")
            
            f.write("约束检查: ")
            checks = []
            checks.append("流量✔" if target_flow <= total_flow <= target_flow + 50 else "流量✖")
            checks.append("效率✔" if ((MIN_TOTAL_EFFICIENCY <= efficiency <= MAX_TOTAL_EFFICIENCY) or efficiency==0.0) else "效率✖")
            checks.append("电耗✔" if ((MIN_KWH_PER_KT <= kwh_per_kt <= MAX_KWH_PER_KT) or kwh_per_kt==0.0) else "电耗✖")
            f.write(", ".join(checks) + "\n")
            f.write("-" * 50 + "\n")
        
        f.write(f"\n=== 结束 ===\n共 {total_combinations_count} 个组合\n")

# ====== 3. 暴力搜索优化器 ======
class BruteForcePumpOptimizer:
    def __init__(self, min_total_efficiency, max_total_efficiency):
        # 从命令行传入的总管效率上下限
        self.MIN_TOTAL_EFFICIENCY = min_total_efficiency
        self.MAX_TOTAL_EFFICIENCY = max_total_efficiency

    def generate_pump_candidates(self, pump_idx, target_pressure, water_level):
        """为单个泵生成频率候选点"""
        # 先检查该泵在当前压力下是否可运行
        if not check_pump_pressure_valid(pump_idx, target_pressure):
            return []  # 不可运行则返回空列表
        
        candidates = []
        for freq in VALID_FREQUENCIES:
            flow = calculate_pump_flow_3d(pump_idx, freq, target_pressure)
            input_power = calculate_pump_input_power(pump_idx, freq,target_pressure,flow)
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
        """寻找最优泵组合（以总管效率最高为最优标准）"""
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
        
        # 步骤2：搜索最优组合（以总管效率最高为标准）
        best_solution = None
        best_total_efficiency = -1.0  # 初始化为-1，确保任何有效解都会更新
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
            
            # 计算当前组合的各项指标
            total_flow = 0.0
            total_input_power = 0.0
            total_output_power = 0.0
            valid = True
            pump_details = []
            
            for candidate in candidate_combination:
                # 检查单泵效率约束
                if candidate['效率'] > MAX_PUMP_EFFICIENCY or candidate['效率'] < MIN_PUMP_EFFICIENCY:
                    valid = False
                    break
                    
                pump_idx = candidate['泵编号'] - 1
                rated_flow = PUMP_RATED_FLOWS[pump_idx]

                # 检查流量约束
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
            
            # 检查流量偏差
            if not (target_flow <= total_flow <= (target_flow + 50)):
                continue
            
            # 计算总管效率和千吨水电耗
            total_efficiency = calculate_total_efficiency(total_output_power, total_input_power)
            kwh_per_kt = calculate_kwh_per_kt(total_input_power, total_flow)
            
            # 检查总管效率约束和千吨水电耗约束
            if not (self.MIN_TOTAL_EFFICIENCY <= total_efficiency <= self.MAX_TOTAL_EFFICIENCY):
                continue
            if not (MIN_KWH_PER_KT <= kwh_per_kt <= MAX_KWH_PER_KT):
                continue
            
            # 以总管效率最高为最优标准
            if total_efficiency > best_total_efficiency:
                best_total_efficiency = total_efficiency
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
            print(f"最优解总管效率: {best_solution['总管效率']} (总功率: {best_solution['总输入功率(kW)']} kW)")
        else:
            save_failed_combinations(target_pressure, water_level, target_flow, 
                            pump_status, active_pumps, all_candidates, total_combinations)
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
    
    parser.add_argument('--min-eff',
                        type=float,
                        default=0.4,
                        help='总管效率下限 (默认: 0.4)')
    parser.add_argument('--max-eff',
                        type=float,
                        default=0.9,
                        help='总管效率上限 (默认: 0.9)')

    # 解析参数
    args = parser.parse_args()
    MIN_TOTAL_EFFICIENCY = args.min_eff
    MAX_TOTAL_EFFICIENCY = args.max_eff


    # 遍历参数范围
    pump_combinations = generate_pump_combinations()
    pump_combinations = [[0, 0, 1, 0, 1]]
    print("="*80)
    print("泵组优化计算程序（以总管效率最高为最优标准）")
    print("="*80)
    print(f"当前配置的总管效率约束：{MIN_TOTAL_EFFICIENCY} ~ {MAX_TOTAL_EFFICIENCY}")
    print(f"最优解标准：总管效率最高")
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
                    if formatted_result:
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
                    '进度': f"{len(all_results)}/{total_combs}"
                })

    # 保存结果
    end_total_time = time.time()
    actual_total_sec = end_total_time - start_total_time

    # 直接使用有效结果列表生成DataFrame（无空值）
    df_results = pd.DataFrame(all_results)
    if not df_results.empty:
        # 更新列顺序，包含新增的泵出口流量字段
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
    output_file = f"计算最优策略_效率{MIN_TOTAL_EFFICIENCY}-{MAX_TOTAL_EFFICIENCY}_{time_str}.csv"
    # df_results.to_csv(output_file, index=False, encoding='utf-8-sig')

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