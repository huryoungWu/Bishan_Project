import pandas as pd
import numpy as np
from itertools import product


def check_missing_combinations(csv_path):
    """
    检查CSV文件中是否遗漏指定参数的排列组合

    参数:
    csv_path: CSV文件的完整路径
    """
    # ====================== 1. 定义参数范围和步长 ======================
    # Flow_Lower: 起始值，结束值，步长
    flow_start, flow_end, flow_step = 1000, 8650, 50
    # Header_Press: 起始值，结束值，步长
    press_start, press_end, press_step = 0.35, 0.59, 0.01
    # Liquid_Level: 起始值，结束值，步长
    #level_start, level_end, level_step = 3.5, 6.5, 0.5

    # 生成所有理论上的参数值（处理浮点数精度问题）
    flow_values = np.arange(flow_start, flow_end + flow_step, flow_step).tolist()
    press_values = np.round(np.arange(press_start, press_end + press_step, press_step), 2).tolist()
    #level_values = np.round(np.arange(level_start, level_end + level_step, level_step), 1).tolist()

    # 生成所有排列组合
    theoretical_combinations = set(product(
        flow_values,
        press_values
        #level_values
    ))

    print(f"理论总组合数: {len(theoretical_combinations)}")

    # ====================== 2. 读取CSV文件 ======================
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        # 尝试其他编码
        df = pd.read_csv(csv_path, encoding='gbk')

    # 检查必要的列是否存在
    #required_columns = ['Flow_Lower', 'Header_Press', 'Liquid_Level']
    required_columns = ['Flow_Lower', 'Header_Press']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件中缺少必要的列: {col}")

    # 处理数据精度问题，避免因小数位数导致的匹配错误
    df['Flow_Lower'] = df['Flow_Lower'].round(0)  # 整数
    df['Header_Press'] = df['Header_Press'].round(2)  # 两位小数
    #df['Liquid_Level'] = df['Liquid_Level'].round(1)  # 一位小数

    # 提取实际存在的组合
    actual_combinations = set(
        df[required_columns].itertuples(index=False, name=None)
    )

    print(f"CSV中实际存在的组合数: {len(actual_combinations)}")

    # ====================== 3. 对比找出遗漏的组合 ======================
    missing_combinations = theoretical_combinations - actual_combinations

    # ====================== 4. 输出结果 ======================
    if not missing_combinations:
        print("\n✅ 恭喜！所有组合都存在，没有遗漏")
    else:
        print(f"\n❌ 发现 {len(missing_combinations)} 个遗漏的组合：")

        # 将遗漏的组合整理成DataFrame，方便查看
        missing_df = pd.DataFrame(
            sorted(missing_combinations),
            columns=required_columns
        )

        # 显示前20个遗漏项（避免输出过多）
        print("\n前20个遗漏的组合：")
        print(missing_df.head(20))

        # 将所有遗漏的组合保存到CSV文件
        missing_csv_path = csv_path.replace('.csv', '_遗漏组合.csv')
        missing_df.to_csv(missing_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n📄 所有遗漏的组合已保存到：{missing_csv_path}")

    return missing_combinations


# ====================== 主程序 ======================
if __name__ == "__main__":
    # CSV文件路径
    csv_path = r"计算最优策略_流量范围550_20260103_064648.csv"

    try:
        # 执行检查
        missing = check_missing_combinations(csv_path)
    except Exception as e:
        print(f"程序执行出错：{e}")