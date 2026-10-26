import pandas as pd
def round_to_nearest(value, base):
    """四舍五入到指定基数的倍数"""
    if pd.isna(value):
        return value
    else:
        if round(value / base) * base > value:
            return round(value / base) * base - base
        else:
            return round(value / base) * base
   
print(round_to_nearest(1916, 50))