import numpy as np

# ====== 常量定义 ======
DENSITY_WATER = 1000  # kg/m³
GRAVITY = 9.81        # m/s²
PRESSURE_TO_HEAD = 102  # 1MPa = 102m扬程
EFFICIENCY_DENOMINATOR = 3600000  # 输出功率计算分母
FLOW_EPS = 0  # 流量极小值阈值
POWER_EPS = 0  # 功率极小值阈值

# 泵额定功率 (kW) 和额定流量 (m³/h)
PUMP_RATED_POWER = np.array([315, 315, 315, 315, 710])  # 转为数组，支持向量化
PUMP_RATED_FLOWS = np.array([1460, 1460, 1460, 1460, 2910])  # 各泵额定流量
POWER_FACTORS = [0.89, 0.89, 0.86, 0.89, 0.83]  # 各泵功率因子

# 千吨水电耗范围
MIN_KWH_PER_KT = 100.0
MAX_KWH_PER_KT = 200.0

# 泵效率合理范围
MIN_PUMP_EFFICIENCY = 0.4
MAX_PUMP_EFFICIENCY = 0.9

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

def check_pump_flow_exceeds_rated(pump_idx, flow):
    """检查泵流量是否超过额定流量"""
    return flow > PUMP_RATED_FLOWS[pump_idx]

def check_kwh_per_kt_in_range(kwh_per_kt):
    """检查千吨水电耗是否在100-200之间"""
    return MIN_KWH_PER_KT <= kwh_per_kt <= MAX_KWH_PER_KT

def check_pump_efficiency_in_range(efficiency):
    """检查单个泵效率是否等于0或在0.4-0.9之间"""
    return abs(efficiency) == 0 or (MIN_PUMP_EFFICIENCY <= efficiency <= MAX_PUMP_EFFICIENCY)

def calculate_pump_performance(target_pressure, water_level, target_flow, pump_frequencies):
    """
    计算泵组性能
    
    参数:
        target_pressure: 总管压力(MPa)
        water_level: 吸水井液位(m)
        target_flow: 目标流量(m³/h)
        pump_frequencies: 各泵频率列表(Hz), 长度应为5(对应5个泵)
    
    返回:
        dict: 包含计算结果和检查结果
    """
    if len(pump_frequencies) != 5:
        raise ValueError("泵频率列表长度必须为5")
    
    total_flow = 0.0
    total_input_power = 0.0
    total_output_power = 0.0
    
    pump_details = []
    flow_exceeds_rated = []  # 记录哪些泵流量超过额定值
    efficiency_out_of_range = []  # 记录哪些泵效率超出合理范围
    
    for pump_idx, freq in enumerate(pump_frequencies):
        # 计算单个泵的流量
        flow = calculate_pump_flow_3d(pump_idx, freq, target_pressure)
        
        # 检查流量是否超过额定值
        if flow > 0 and check_pump_flow_exceeds_rated(pump_idx, flow):
            flow_exceeds_rated.append(pump_idx + 1)  # 记录泵编号(1-5)
        
        # 计算输入功率
        input_power = calculate_pump_input_power(pump_idx, freq)
        
        # 计算输出功率
        output_power = calculate_single_pump_output_power(flow, target_pressure, water_level)
        
        # 计算效率
        efficiency = output_power / input_power if input_power > 0 else 0.0
        
        # 检查效率是否在合理范围内
        if flow > 0 and not check_pump_efficiency_in_range(efficiency):
            efficiency_out_of_range.append(pump_idx + 1)  # 记录泵编号(1-5)
        
        # 累加总和
        total_flow += flow
        total_input_power += input_power
        total_output_power += output_power
        
        # 保存单泵详情
        pump_details.append({
            '泵编号': pump_idx + 1,
            '频率(Hz)': freq,
            '出口流量(m³/h)': round(flow, 2),
            '输入功率(kW)': round(input_power, 2),
            '输出功率(kW)': round(output_power, 2),
            '效率': round(efficiency, 4),
            '超过额定流量': flow > PUMP_RATED_FLOWS[pump_idx],
            '效率是否合理': check_pump_efficiency_in_range(efficiency)
        })
    
    # 计算总管效率
    efficiency = calculate_total_efficiency(total_output_power, total_input_power)
    
    # 计算千吨水电耗
    kwh_per_kt = calculate_kwh_per_kt(total_input_power, total_flow)
    
    # 检查千吨水电耗是否在合理范围内
    kwh_in_range = check_kwh_per_kt_in_range(kwh_per_kt)
    
    return {
        '总管压力(MPa)': target_pressure,
        '吸水井液位(m)': water_level,
        '目标流量(m³/h)': target_flow,
        '实际总流量(m³/h)': round(total_flow, 2),
        '流量偏差(m³/h)': round(total_flow - target_flow, 2),
        '总输入功率(kW)': round(total_input_power, 2),
        '总输出功率(kW)': round(total_output_power, 2),
        '总管效率': efficiency,
        '千吨水电耗(kWh/kt)': kwh_per_kt,
        '千吨水电耗是否合理': kwh_in_range,
        '流量超限泵列表': flow_exceeds_rated,
        '效率异常泵列表': efficiency_out_of_range,
        '单泵详情': pump_details
    }

def print_results(result):
    """格式化打印计算结果"""
    print("\n=== 泵组性能计算结果 ===")
    print(f"总管压力: {result['总管压力(MPa)']} MPa")
    print(f"吸水井液位: {result['吸水井液位(m)']} m")
    print(f"目标流量: {result['目标流量(m³/h)']} m³/h")
    print(f"实际总流量: {result['实际总流量(m³/h)']} m³/h (偏差: {result['流量偏差(m³/h)']} m³/h)")
    print(f"总输入功率: {result['总输入功率(kW)']} kW")
    print(f"总输出功率: {result['总输出功率(kW)']} kW")
    print(f"总管效率: {result['总管效率']:.2%}")
    print(f"千吨水电耗: {result['千吨水电耗(kWh/kt)']} kWh/kt")
    
    # 新增检查结果输出
    print("\n=== 检查结果 ===")
    if result['千吨水电耗是否合理']:
        print("✅ 千吨水电耗在合理范围内 (100-200 kWh/kt)")
    else:
        print(f"⚠️ 千吨水电耗超出合理范围: {result['千吨水电耗(kWh/kt)']} kWh/kt")
    
    if result['流量超限泵列表']:
        print(f"⚠️ 以下泵的流量超过额定值: {', '.join(map(str, result['流量超限泵列表']))}")
    else:
        print("✅ 所有泵的流量均在额定值范围内")
    
    if result['效率异常泵列表']:
        print(f"⚠️ 以下泵的效率不在0或0.4-0.9范围内: {', '.join(map(str, result['效率异常泵列表']))}")
    else:
        print("✅ 所有泵的效率均在0或0.4-0.9范围内")
    
    print("\n=== 单泵运行详情 ===")
    for pump in result['单泵详情']:
        flow_status = "⚠️超限" if pump['超过额定流量'] else "✅正常"
        eff_status = "⚠️异常" if not pump['效率是否合理'] else "✅正常"
        print(f"泵{pump['泵编号']} (流量:{flow_status}, 效率:{eff_status}):")
        print(f"  频率: {pump['频率(Hz)']} Hz")
        print(f"  出口流量: {pump['出口流量(m³/h)']} m³/h (额定: {PUMP_RATED_FLOWS[pump['泵编号']-1]} m³/h)")
        print(f"  输入功率: {pump['输入功率(kW)']} kW")
        print(f"  输出功率: {pump['输出功率(kW)']} kW")
        print(f"  效率: {pump['效率']:.2%}")
        print()

if __name__ == "__main__":
    # 示例输入参数
    target_pressure = 0.4  # MPa
    water_level = 3 # m
    target_flow = 2000  # m³/h
    pump_frequencies = [0.0, 0.0, 40.0, 0.0, 40.0]  # 泵1-5的频率(Hz)
    
    # 计算性能
    result = calculate_pump_performance(target_pressure, water_level, target_flow, pump_frequencies)
    
    # 打印结果
    print_results(result)
