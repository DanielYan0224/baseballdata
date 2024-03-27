#%%
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import fractions
import itertools
import collections
from fractions import Fraction as F
# 讀取 CSV 檔案
data_2022 = pd.read_csv(r"C:\Users\user\Desktop\baseballdata\data2022.csv")
data_2022.insert(1, 'year', '2022')

data_2023 = pd.read_csv(r"C:\Users\user\Desktop\baseballdata\data2023.csv")
data_2023.insert(1, 'year', '2023')

cole_2223 = merged_data = pd.concat([data_2022, data_2023], axis=0)
cole_2223.to_csv(r"C:\Users\user\Desktop\baseballdata\cole_2223.csv")

cole_2223 = cole_2223[cole_2223['pitch_type'] == 'SL']

year = '2022'
cole_2223 = cole_2223[(cole_2223['description'] == 'hit_into_play')\
            & (cole_2223['year'] == year)]

x_label = 'zone'
y_label = 'effective_speed'


# 定義條件以更改顏色
condition = (cole_2223['events'] == 'field_out')
condition_reverse = ~condition  # 反轉條件

# 將符合特定條件的數據點移至數據框頂部
data_sorted = pd.concat([cole_2223[condition], cole_2223[~condition]])

fig = px.scatter(data_sorted, x=x_label, y=y_label, color=condition, \
    color_discrete_map={True: 'red', False: 'gray'}, 
symbol=condition, marginal_x="histogram", marginal_y="histogram",
title=f'Scattorplot of {year}')

# fig.update_xaxes(title_text = x_label)
# fig.update_yaxes(title_text = y_label)
# fig.update_layout(title= year, title_x=0.5, title_y=0.95)
bins_x = [cole_2223['launch_angle'].min(), -20, \
        0, 30, 50, cole_2223['launch_angle'].max()]
bins_y = [cole_2223['launch_speed'].min(), 70, \
        80, 100, cole_2223['launch_speed'].max()]


# graph line on x-axis
# for i in range(1, len(bins_x)):
#     fig.add_shape(type="line", 
#             x0=bins_x[i], y0=0, x1=bins_x[i], y1=1, 
#             xref="x", yref="paper",  
#             line=dict(color="RoyalBlue", width=1))

# # graph line on y-axis
# bins_y = [cole_2223['launch_speed'].min(), 70, \
#         80, 100, cole_2223['launch_speed'].max()]
# for i in range(1, len(bins_y)):
#     fig.add_shape(type="line", 
#             x0=0, y0=bins_y[i], x1=1, y1=bins_y[i], 
#             xref="paper", yref="y",  
#             line=dict(color="RoyalBlue", width=1))


fig.write_html('scatter_plot.html', auto_open=True)


#%%
