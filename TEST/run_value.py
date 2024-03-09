#%%
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def run_expectancy(df, level='plate appearence'):

    # 排序
    df.cort_values(by = by=['game_pk', 'at_bat_number', 'pitch_number'], inplace=True)

    # last pitch
    df['last_pitch_at_bat'] = df.groupby(['game_pk', 'at_bat_number', 'inning_topbot'])\
    ['pitch_number'].transform(lambda x: (x == x.max()).astype(int))

    print
#%%