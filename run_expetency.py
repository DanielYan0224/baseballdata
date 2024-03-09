#%%
import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\閻天立\Desktop\pybaseball\data2019.csv")

# 簡化數據集，保留關鍵列
simplified_data = data[['outs_when_up', 'on_1b', 'on_2b', 'on_3b', 'delta_run_exp']].copy()

# 標記壘包狀態
simplified_data['base_state'] = (
    simplified_data[['on_1b', 'on_2b', 'on_3b']].notnull().astype(int).astype(str).agg('-'.join, axis=1)
)

# 分組計算每種壘包狀態和出局數下的平均跑點預期變化
grouped_run_exp = simplified_data.groupby(['outs_when_up', 'base_state'])['delta_run_exp'].mean().reset_index()

# 顯示計算結果
grouped_run_exp



#%%