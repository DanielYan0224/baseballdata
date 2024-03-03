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

year = '2023'
cole_2223 = cole_2223[cole_2223['year'] == year]


# 提取 x 和 y 變量
x_label = 'plate_z'
y_label = 'launch_speed'
x = cole_2223[x_label]
y = cole_2223[y_label]

# 定義條件以更改顏色
condition_gray = (cole_2223['events'] == 'field_out')
condition_black = ~condition_gray  # 反轉條件

# 將符合特定條件的數據點移至數據框頂部
data_sorted = pd.concat([cole_2223[condition_gray], cole_2223[~condition_gray]])

fig = px.scatter(data_sorted, x=x, y=y, color=condition_gray, color_discrete_map={True: 'red', False: 'gray'}, 
                symbol=condition_gray, symbol_map={True: 'diamond', False: 'circle'})  


fig.update_xaxes(title_text = x_label)
fig.update_yaxes(title_text = y_label)
fig.update_layout(title= year, title_x=0.5, title_y=0.95)

# 將散點圖儲存為 HTML
fig.write_html('scatter_plot.html', auto_open=True)


#%%
