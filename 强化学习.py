import json
import numpy as np
import pandas as pd
import time
import multiprocessing
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ====== 新增：强化学习相关库 ======
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# ====== 0. 全局配置与常量 ======
PRESSURE_TOLERANCE = 0.01  # 压力波动容差
FLOW_TOLERANCES = [25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]  # 逐步放宽的流量限制值

# RL超参数
RL_CONFIG = {
    "EPISODES": 200,          # 训练轮数
    "BATCH_SIZE": 64,         # 批次大小
    "GAMMA": 0.95,            # 折扣因子
    "EPSILON": 1.0,           # 探索率初始值
    "EPSILON_MIN": 0.01,      # 探索率最小值
    "EPSILON_DECAY": 0.995,   # 探索率衰减率
    "LEARNING_RATE": 0.001,   # 学习率
    "MEMORY_CAPACITY": 10000  # 经验回放池容量
}

# ====== 新增：RL网络定义 ======
class DQN(nn.Module):
    """深度Q网络"""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """DQN智能体"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=RL_CONFIG["MEMORY_CAPACITY"])
        
        # 网络初始化
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=RL_CONFIG["LEARNING_RATE"])
        self.criterion = nn.MSELoss()
        
        # 超参数
        self.gamma = RL_CONFIG["GAMMA"]
        self.epsilon = RL_CONFIG["EPSILON"]
        self.epsilon_min = RL_CONFIG["EPSILON_MIN"]
        self.epsilon_decay = RL_CONFIG["EPSILON_DECAY"]
        
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """选择动作（ε-贪心策略）"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()
    
    def replay(self, batch_size):
        """经验回放训练"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)[0]).item()
            
            target_f = self.model(state)
            target_f[0][action] = target
            
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ====== 泵组优化RL环境 ======
class PumpEnv:
    """泵组优化强化学习环境"""
    def __init__(self, pump_candidates, target_flow, tolerance, constraints, phys):
        self.pump_candidates = pump_candidates
        self.target_flow = target_flow
        self.tolerance = tolerance
        self.constraints = constraints
        self.phys = phys
        
        # 每个泵的可选动作数
        self.pump_action_sizes = [len(c['frequencies']) for c in pump_candidates]
        self.total_action_size = np.prod(self.pump_action_sizes)
        
        # 状态空间：目标流量、当前总流量、当前总功率、流量偏差
        self.state_size = 4
        
        # 初始化状态
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.current_indices = [0] * len(self.pump_candidates)
        self.current_step = 0
        self.done = False
        
        # 初始状态
        flows = [self.pump_candidates[i]['flows'][idx] for i, idx in enumerate(self.current_indices)]
        input_powers = [self.pump_candidates[i]['input_powers'][idx] for i, idx in enumerate(self.current_indices)]
        total_flow = np.sum(flows)
        total_power = np.sum(input_powers)
        flow_error = abs(total_flow - self.target_flow)
        
        return [self.target_flow, total_flow, total_power, flow_error]
    
    def _indices_to_action(self, indices):
        """索引组合转动作编号"""
        action = 0
        multiplier = 1
        for i in reversed(range(len(indices))):
            action += indices[i] * multiplier
            multiplier *= self.pump_action_sizes[i]
        return action
    
    def _action_to_indices(self, action):
        """动作编号转索引组合"""
        indices = []
        remaining = action
        for size in reversed(self.pump_action_sizes):
            indices.append(remaining % size)
            remaining = remaining // size
        return list(reversed(indices))
    
    def step(self, action):
        """执行动作"""
        # 转换动作到索引
        self.current_indices = self._action_to_indices(action)
        
        # 获取当前参数
        flows = []
        input_powers = []
        output_powers = []
        for i, idx in enumerate(self.current_indices):
            flows.append(self.pump_candidates[i]['flows'][idx])
            input_powers.append(self.pump_candidates[i]['input_powers'][idx])
            output_powers.append(self.pump_candidates[i]['output_powers'][idx])
        
        # 计算总参数
        total_flow = np.sum(flows)
        total_input = np.sum(input_powers)
        total_output = np.sum(output_powers)
        
        # 计算奖励
        reward = 0
        flow_error = abs(total_flow - self.target_flow)
        
        # 基础奖励：流量偏差越小奖励越高
        if flow_error <= self.tolerance:
            reward += 100 - flow_error
        else:
            reward -= flow_error
        
        # 功率奖励：功率越小奖励越高
        reward += max(0, 1000 - total_input) / 10
        
        # 效率约束奖励
        if total_input > self.phys['POWER_EPS'] and total_output > self.phys['POWER_EPS']:
            total_efficiency = total_output / total_input
            if (self.constraints['MIN_TOTAL_EFFICIENCY'] <= total_efficiency <= 
                self.constraints['MAX_TOTAL_EFFICIENCY']):
                reward += 50
            else:
                reward -= 50
        
        # 千吨水电耗约束奖励
        if total_flow > self.phys['FLOW_EPS']:
            kwh_per_kt = total_input / (total_flow / 1000)
            if (self.constraints['MIN_KWH_PER_KT'] <= kwh_per_kt <= 
                self.constraints['MAX_KWH_PER_KT']):
                reward += 50
            else:
                reward -= 50
        
        # 终止条件
        self.current_step += 1
        if self.current_step >= 100 or flow_error <= self.tolerance:
            self.done = True
        
        # 新状态
        next_state = [self.target_flow, total_flow, total_input, flow_error]
        
        return next_state, reward, self.done, {
            'indices': self.current_indices,
            'total_flow': total_flow,
            'total_input': total_input,
            'total_output': total_output
        }

# ====== 1. 配置加载与预处理 ======
def load_config(config_path='filling_condition_config.json'):
    """加载并预处理配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 物理常量
    phys_const = {
        'DENSITY_WATER': config['physical_constants']['DENSITY_WATER'],
        'GRAVITY': config['physical_constants']['GRAVITY'],
        'PRESSURE_TO_HEAD': config['physical_constants']['PRESSURE_TO_HEAD'],
        'EFFICIENCY_DENOMINATOR': config['physical_constants']['EFFICIENCY_DENOMINATOR'],
        'FLOW_EPS': config['physical_constants']['FLOW_EPS'],
        'POWER_EPS': config['physical_constants']['POWER_EPS']
    }
    
    # 泵参数
    pump_specs = {
        'RATED_POWER': np.array(config['pump_specs']['PUMP_RATED_POWER']),
        'RATED_FLOWS': np.array(config['pump_specs']['PUMP_RATED_FLOWS']),
        'POWER_FACTORS': np.array(config['pump_specs']['POWER_FACTORS']),
        'PRESSURE_RANGE': np.array([
            config['pump_pressure_ranges'][str(i)] for i in range(5)
        ], dtype=np.float64),
        'FLOW_COEFFS': np.array([
            config['pump_3d_flow_func'][str(i)]['coeffs'] for i in range(5)
        ], dtype=np.float64)
    }
    
    # 约束条件
    constraints = {
        'MAX_FLOW_DEVIATION_ABS': config['constraints']['MAX_FLOW_DEVIATION_ABS'],
        'MIN_PUMP_EFFICIENCY': config['constraints']['MIN_PUMP_EFFICIENCY'],
        'MAX_PUMP_EFFICIENCY': config['constraints']['MAX_PUMP_EFFICIENCY'],
        'MIN_KWH_PER_KT': config['constraints']['MIN_KWH_PER_KT'],
        'MAX_KWH_PER_KT': config['constraints']['MAX_KWH_PER_KT'],
        'MIN_TOTAL_EFFICIENCY': config['optimization_params']['MIN_TOTAL_EFFICIENCY'],
        'MAX_TOTAL_EFFICIENCY': config['optimization_params']['MAX_TOTAL_EFFICIENCY']
    }
    
    # 计算范围
    calc_ranges = config['calculation_ranges']
    ranges = {
        'TARGET_FLOWS': np.arange(
            calc_ranges['TARGET_FLOWS']['start'],
            calc_ranges['TARGET_FLOWS']['end'],
            calc_ranges['TARGET_FLOWS']['step']
        ).astype(np.float64),
        'TARGET_PRESSURES': np.arange(
            calc_ranges['TARGET_PRESSURES']['start'],
            calc_ranges['TARGET_PRESSURES']['end'],
            calc_ranges['TARGET_PRESSURES']['step']
        ).astype(np.float64),
        'WATER_LEVELS': np.arange(
            calc_ranges['WATER_LEVELS']['start'],
            calc_ranges['WATER_LEVELS']['end'],
            calc_ranges['WATER_LEVELS']['step']
        ).astype(np.float64),
        'VALID_FREQUENCIES': np.arange(
            calc_ranges['VALID_FREQUENCIES']['start'],
            calc_ranges['VALID_FREQUENCIES']['end'],
            calc_ranges['VALID_FREQUENCIES']['step']
        ).astype(np.float64)
    }
    
    return {
        'physical': phys_const,
        'pump': pump_specs,
        'constraints': constraints,
        'ranges': ranges
    }

# ====== 2. 核心计算函数 ======
def vectorized_pump_flow_3d(pump_idx, frequencies, pressure, coeffs):
    """3D流量函数计算"""
    f = frequencies
    p = np.full_like(f, pressure)
    
    # 预计算幂次
    f2, f3 = f**2, f**3
    p2, p3 = p**2, p**3
    
    # 多项式计算
    flow = (
        coeffs[0] + coeffs[1]*f + coeffs[2]*p + coeffs[3]*f2 + coeffs[4]*p2 +
        coeffs[5]*f*p + coeffs[6]*f3 + coeffs[7]*p3 + coeffs[8]*f2*p + coeffs[9]*f*p2
    )
    flow[flow < 0] = 0.0
    return flow

def vectorized_pump_input_power(pump_idx, frequencies, rated_power, power_factors):
    """输入功率计算"""
    base_power = rated_power[pump_idx] * (frequencies / 50) ** 3
    return base_power * power_factors[pump_idx]

def vectorized_pump_calculations(pump_idx, pressure, water_level, config):
    """单泵参数批量计算"""
    phys = config['physical']
    pump_cfg = config['pump']
    constraints = config['constraints']
    frequencies = config['ranges']['VALID_FREQUENCIES']
    
    # 压力候选点
    pressure_candidates = np.array([
        pressure - PRESSURE_TOLERANCE,
        pressure,
        pressure + PRESSURE_TOLERANCE
    ])
    
    # 过滤有效压力点
    valid_pressures = []
    for p in pressure_candidates:
        min_p, max_p = pump_cfg['PRESSURE_RANGE'][pump_idx]
        if min_p <= p <= max_p:
            valid_pressures.append(p)
    
    if not valid_pressures:
        return None
    
    # 选择最接近目标压力的压力值
    selected_pressure = min(valid_pressures, key=lambda p: abs(p - pressure))
    
    # 批量计算所有频率的参数
    flows = vectorized_pump_flow_3d(
        pump_idx, frequencies, selected_pressure, pump_cfg['FLOW_COEFFS'][pump_idx]
    )
    input_powers = vectorized_pump_input_power(
        pump_idx, frequencies, pump_cfg['RATED_POWER'], pump_cfg['POWER_FACTORS']
    )
    
    # 计算输出功率
    H = selected_pressure * phys['PRESSURE_TO_HEAD']
    effective_head = H - (water_level - 2.35)
    output_powers = (
        phys['DENSITY_WATER'] * phys['GRAVITY'] * flows * effective_head
    ) / phys['EFFICIENCY_DENOMINATOR']
    
    # 计算效率并过滤
    efficiencies = (output_powers / input_powers) * 100
    efficiencies[input_powers < phys['POWER_EPS']] = 0.0
    
    # 过滤有效候选
    valid_mask = (
        (efficiencies >= constraints['MIN_PUMP_EFFICIENCY']) &
        (efficiencies <= constraints['MAX_PUMP_EFFICIENCY']) &
        (flows >= phys['FLOW_EPS'])
    )
    
    if not np.any(valid_mask):
        return None
    
    # 构造候选数据
    return {
        'pump_idx': pump_idx,
        'frequencies': frequencies[valid_mask],
        'flows': flows[valid_mask],
        'input_powers': input_powers[valid_mask],
        'output_powers': output_powers[valid_mask],
        'efficiencies': efficiencies[valid_mask]
    }

# ====== 3. 优化器类 ======
class OptimizerWorker:
    """优化器工作进程"""
    def __init__(self, config):
        self.config = config
        self.phys = config['physical']
        self.constraints = config['constraints']
    
    def is_possible_combination(self, pump_status, target_flow, target_pressure, tolerance):
        """带流量限制的组合有效性检查"""
        pump_cfg = self.config['pump']
        active_pumps = np.where(np.array(pump_status) == 1)[0]
        
        if len(active_pumps) == 0:
            return False
        
        # 压力检查
        pressure_ranges = pump_cfg['PRESSURE_RANGE'][active_pumps]
        adjusted_pressure_valid = (
            ((target_pressure - PRESSURE_TOLERANCE) <= pressure_ranges[:, 1]) &
            ((target_pressure + PRESSURE_TOLERANCE) >= pressure_ranges[:, 0])
        )
        
        if not np.all(adjusted_pressure_valid):
            return False
        
        # 流量范围检查
        total_min_flow = (
            np.sum(pump_cfg['RATED_FLOWS'][active_pumps]) * 0.8 - 
            tolerance
        )
        total_max_flow = (
            np.sum(pump_cfg['RATED_FLOWS'][active_pumps]) + 
            tolerance
        )
        
        return (target_flow >= total_min_flow) & (target_flow <= total_max_flow)
    
    def find_optimal_combination(self, task_args):
        """单任务优化计算（使用强化学习替代暴力搜索）"""
        pump_status, target_flow, target_pressure, water_level, combo_idx = task_args
        start_time = time.time()
        best_solution = None
        final_tolerance = None
        
        # 尝试不同的流量限制
        for tolerance in FLOW_TOLERANCES:
            # 快速过滤无效组合
            if not self.is_possible_combination(pump_status, target_flow, target_pressure, tolerance):
                continue
            
            # 提取运行泵
            active_pumps = np.where(np.array(pump_status) == 1)[0]
            if len(active_pumps) == 0:
                continue
            
            # 批量计算所有运行泵的有效候选
            pump_candidates = []
            for pump_idx in active_pumps:
                candidate = vectorized_pump_calculations(
                    pump_idx, target_pressure, water_level, self.config
                )
                if candidate is None:
                    continue
                pump_candidates.append(candidate)
            
            if not pump_candidates:
                continue
            
            # ====== 核心修改：使用强化学习替代暴力搜索 ======
            # 创建RL环境
            env = PumpEnv(
                pump_candidates=pump_candidates,
                target_flow=target_flow,
                tolerance=tolerance,
                constraints=self.constraints,
                phys=self.phys
            )
            
            # 创建智能体
            state_size = env.state_size
            action_size = env.total_action_size
            
            # 跳过动作空间过大的情况
            if action_size > 100000:
                continue
                
            agent = DQNAgent(state_size, action_size)
            
            # 训练智能体
            best_reward = -np.inf
            best_result = None
            
            for e in range(RL_CONFIG["EPISODES"]):
                state = env.reset()
                total_reward = 0
                
                while not env.done:
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    
                    state = next_state
                    total_reward += reward
                    
                    if done:
                        # 检查是否满足所有约束
                        total_input = info['total_input']
                        total_output = info['total_output']
                        total_flow = info['total_flow']
                        indices = info['indices']
                        
                        # 总管效率计算
                        if (total_input > self.phys['POWER_EPS'] and 
                            total_output > self.phys['POWER_EPS']):
                            total_efficiency = total_output / total_input
                            
                            # 效率约束检查
                            if (self.constraints['MIN_TOTAL_EFFICIENCY'] <= total_efficiency <= 
                                self.constraints['MAX_TOTAL_EFFICIENCY']):
                                
                                # 千吨水电耗计算
                                if total_flow > self.phys['FLOW_EPS']:
                                    kwh_per_kt = total_input / (total_flow / 1000)
                                    
                                    # 电耗约束检查
                                    if (self.constraints['MIN_KWH_PER_KT'] <= kwh_per_kt <= 
                                        self.constraints['MAX_KWH_PER_KT']):
                                        
                                        # 更新最优结果
                                        if (total_reward > best_reward or 
                                            (total_reward == best_reward and 
                                             total_input < (best_result['total_input'] if best_result else np.inf))):
                                            best_reward = total_reward
                                            best_result = {
                                                'indices': indices,
                                                'total_flow': total_flow,
                                                'total_input': total_input,
                                                'total_output': total_output,
                                                'total_efficiency': total_efficiency,
                                                'kwh_per_kt': kwh_per_kt,
                                                'reward': total_reward
                                            }
                        break
                
                # 经验回放
                agent.replay(RL_CONFIG["BATCH_SIZE"])
            
            # 构建最优解
            if best_result:
                pump_details = []
                for i, idx in enumerate(best_result['indices']):
                    pump_idx = active_pumps[i]
                    pump_details.append({
                        '泵编号': pump_idx + 1,
                        '频率(Hz)': pump_candidates[i]['frequencies'][idx],
                        '出口流量': pump_candidates[i]['flows'][idx],
                        '输入功率(kW)': pump_candidates[i]['input_powers'][idx],
                        '输出功率(kW)': pump_candidates[i]['output_powers'][idx],
                        '效率(%)': pump_candidates[i]['efficiencies'][idx]
                    })
                
                current_best_solution = {
                    '总流量': round(best_result['total_flow'], 2),
                    '总输入功率(kW)': round(best_result['total_input'], 2),
                    '总输出功率(kW)': round(best_result['total_output'], 4),
                    '总管效率(%)': round(best_result['total_efficiency'] * 100, 2),
                    '千吨水电耗(kWh/kt)': round(best_result['kwh_per_kt'], 2),
                    '单泵详情': pump_details,
                    '泵状态': pump_status,
                    '流量限制': tolerance
                }
                
                # 更新最佳解
                if (best_solution is None or 
                    current_best_solution['总输入功率(kW)'] < best_solution['总输入功率(kW)']):
                    best_solution = current_best_solution
                    final_tolerance = tolerance
        
        return {
            'combo_idx': combo_idx,
            'solution': best_solution,
            'elapsed': time.time() - start_time,
            'params': (target_pressure, target_flow, water_level, pump_status),
            'flow_tolerance': final_tolerance,
            'skip_reason': None if best_solution else "未找到可行解"
        }

# ====== 4. 结果格式化 ======
def format_result(solution, params):
    """格式化结果"""
    if not solution:
        return None
    
    target_pressure, target_flow, water_level, pump_status = params
    pump_freqs = np.zeros(6, dtype=np.float64)
    pump_effs = np.zeros(6, dtype=np.float64)
    pump_out_flows = np.zeros(6, dtype=np.float64)

    for pump in solution['单泵详情']:
        pump_id = pump['泵编号'] - 1
        if 0 <= pump_id < 5:
            pump_freqs[pump_id] = pump['频率(Hz)']
            pump_effs[pump_id] = pump['效率(%)']
            pump_out_flows[pump_id] = pump['出口流量']
    
    # 生成运行组合字符串
    run_combo = ''.join(str(s) for s in pump_status) + '0'
    
    return {
        '总管压力': round(target_pressure, 2),
        '总管瞬时流量': round(target_flow, 2),
        '平均吸水井液位': round(water_level, 1),
        '运行组合': run_combo,
        '总管_效率': solution['总管效率(%)'],
        '千吨水电耗_kWh_kt': solution['千吨水电耗(kWh/kt)'],
        '流量限制': solution.get('流量限制', 'N/A'),
        '泵频率给定1': pump_freqs[0],
        '泵频率给定2': pump_freqs[1],
        '泵频率给定3': pump_freqs[2],
        '泵频率给定4': pump_freqs[3],
        '泵频率给定5': pump_freqs[4],
        '泵频率给定6': pump_freqs[5],
        '泵出口流量1': pump_out_flows[0],
        '泵出口流量2': pump_out_flows[1],
        '泵出口流量3': pump_out_flows[2],
        '泵出口流量4': pump_out_flows[3],
        '泵出口流量5': pump_out_flows[4],
        '泵出口流量6': pump_out_flows[5],
        '泵1_效率': pump_effs[0],
        '泵2_效率': pump_effs[1],
        '泵3_效率': pump_effs[2],
        '泵4_效率': pump_effs[3],
        '泵5_效率': pump_effs[4],
        '泵6_效率': pump_effs[5]
    }

# ====== 5. 工具函数 ======
def generate_pump_combinations():
    """生成所有泵组合"""
    return [[(i >> j) & 1 for j in range(5)] for i in range(1, 2**5)]

def format_seconds(seconds):
    """格式化时间"""
    return str(timedelta(seconds=round(seconds)))

# ====== 6. 主函数 ======
def main():
    """主函数"""
    # 1. 加载配置
    print("📌 加载配置文件...")
    config = load_config()
    constraints = config['constraints']
    ranges = config['ranges']
    
    # 2. 生成任务
    print("📌 生成计算任务...")
    pump_combinations = generate_pump_combinations()
    all_task_args = []
    combo_seq = 1
    
    for target_pressure in ranges['TARGET_PRESSURES']:
        for target_flow in ranges['TARGET_FLOWS']:
            for water_level in ranges['WATER_LEVELS']:
                for pump_status in pump_combinations:
                    # 使用最小流量限制进行初步过滤
                    if not OptimizerWorker(config).is_possible_combination(
                        pump_status, target_flow, target_pressure, FLOW_TOLERANCES[0]
                    ):
                        continue
                    all_task_args.append(
                        (pump_status, target_flow, target_pressure, water_level, combo_seq)
                    )
                    combo_seq += 1
    
    total_tasks = len(all_task_args)
    print(f"📌 共生成 {total_tasks} 个有效计算任务")
    print("="*80)
    
    # 3. 初始化
    start_total_time = time.time()
    all_results = []
    valid_solutions = 0
    invalid_solutions = 0
    worker = OptimizerWorker(config)
    max_workers = min(32, multiprocessing.cpu_count())
    
    # 4. 启动进程池
    print(f"🚀 启动 {max_workers} 个进程开始计算...")
    print(f"🕒 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker.find_optimal_combination, args): args for args in all_task_args}
        
        # 进度条
        from tqdm import tqdm
        pbar = tqdm(total=total_tasks, desc="🔧 泵组优化计算", unit="任务")
        
        # 处理结果
        for future in as_completed(futures):
            result = future.result()
            
            if result['solution']:
                valid_solutions += 1
                formatted = format_result(result['solution'], result['params'])
                if formatted:
                    all_results.append(formatted)
            else:
                invalid_solutions += 1
            
            # 更新进度条
            completed = valid_solutions + invalid_solutions
            valid_ratio = (valid_solutions / completed * 100) if completed > 0 else 0
            pbar.set_postfix_str(f"有效解: {valid_solutions} ({valid_ratio:.1f}%) | 无效解: {invalid_solutions}")
            pbar.update(1)
        
        pbar.close()
    
    # 5. 保存结果（修改为保存到Excel）
    end_total_time = time.time()
    total_elapsed = end_total_time - start_total_time
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        column_order = [
            '总管压力', '总管瞬时流量', '平均吸水井液位',
            '运行组合', '总管_效率', '千吨水电耗_kWh_kt', '流量限制',
            '泵频率给定1', '泵频率给定2', '泵频率给定3', '泵频率给定4', '泵频率给定5', '泵频率给定6',
            '泵出口流量1', '泵出口流量2', '泵出口流量3', '泵出口流量4', '泵出口流量5', '泵出口流量6',
            '泵1_效率', '泵2_效率', '泵3_效率', '泵4_效率', '泵5_效率', '泵6_效率'
        ]
        df_results = df_results[column_order].sort_values(
            by=['总管压力', '总管瞬时流量', '平均吸水井液位', '运行组合'],
            ascending=[True, True, True, True]
        ).reset_index(drop=True)
        
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            f"泵组优化结果_效率{constraints['MIN_TOTAL_EFFICIENCY']}-{constraints['MAX_TOTAL_EFFICIENCY']}_"
            f"流量限制{FLOW_TOLERANCES[0]}-{FLOW_TOLERANCES[-1]}_{time_str}.xlsx"
        )
        # 保存为Excel文件
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='泵组优化结果', index=False)
        
        # 输出统计
        valid_ratio = (valid_solutions / total_tasks) * 100 if total_tasks > 0 else 0
        print("\n" + "="*80)
        print("✅ 计算完成！")
        print("="*80)
        print(f"📊 最终统计信息：")
        print(f"   📈 总任务数：{total_tasks}")
        print(f"   ✅ 有效解数量：{valid_solutions} ({valid_ratio:.1f}%)")
        print(f"   ❌ 无效解数量：{invalid_solutions}")
        print(f"   ⏱️  总耗时：{format_seconds(total_elapsed)}")
        print(f"   ⚡ 平均耗时：{total_elapsed/total_tasks:.2f}秒/任务")
        print(f"   📁 结果文件：{output_file}")
        print("="*80)
    else:
        print("\n⚠️ 未找到任何有效解！")
        print(f"🕒 总耗时：{format_seconds(total_elapsed)}")

if __name__ == "__main__":
    # 设置随机种子确保可复现性
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    main()