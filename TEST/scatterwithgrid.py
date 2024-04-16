#%%
import re
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from fractions import Fraction as F
from math import log


#################################
#整理資料
def extract_year(file_path):
    match = re.search(r'\d{4}', file_path)
    if match:
        return int(match.group())  # Convert to integer
    else:
        return None 
    
def load_dataframe(file_path):
    df = pd.read_csv(file_path)
    df.insert(loc=1, column='year', value=extract_year(file_path))
    return df

file_path_1 = r"C:\Users\user\Desktop\baseballdata\2022sppitchingdata.csv"
file_path_2 = r"C:\Users\user\Desktop\baseballdata\2023sppitchingdata.csv"


df1 = load_dataframe(file_path_1)

df2 = load_dataframe(file_path_2)

df = pd.concat([df1, df2], axis=0)

#################################
#可調整
file_path = file_path_1
study_df = load_dataframe(file_path)
x_grids = 40
y_grids = 40
mapp = {'single': 0, 
'double': 0, 
'triple': 0, 
'home_run': 0,
'DP': 1, 
'field_out': 1,
        }
#################################
study_df = study_df[study_df['description'] == 'hit_into_play']
x_label = 'launch_angle'
y_label = 'launch_speed'

# delete the data that has nan in the labels
study_df = study_df.dropna(subset=[x_label, y_label], how='any')

# 只留下安打與出局
def new_event(events):
    mapping = {'single': "single", 'double': "double", 
            'triple': "triple", 'home_run': "home_run", 
            'field_out': "field_out", "double_play": "DP"}
    return mapping.get(events, "others")  # 其他的都是others

study_df['new_event'] = study_df['events'].apply(new_event).astype(str)

symbol_map = {
    'single': 'circle',           
    'double': 'square',           
    'triple': 'diamond',          
    'home_run': 'star',           
    'field_out': 'x',             
    'DP': "hexagon", 
    'others': 'triangle-up'       
}

color_discrete_map = {
    'single': 'blue',
    'double': 'orange',
    'triple': 'green',
    'home_run': 'red',
    'field_out': 'purple',
    'DP': 'black',    
    'others': 'grey'  
}             



#################################
# Draw the scatter plot
fig = px.scatter(study_df, x=x_label, y=y_label, \
        color='new_event', 
        symbol='new_event',
        color_discrete_map=color_discrete_map,
        symbol_map=symbol_map,
        marginal_x="histogram", 
        marginal_y="histogram",
)

# 定義grid
x_min, x_max = df[x_label].min(), df[x_label].max()
y_min, y_max = df[y_label].min(), df[y_label].max()

grid_size_x = (x_max - x_min) / x_grids
grid_size_y = (y_max - y_min) / y_grids
x_lines = np.arange(x_min, x_max, grid_size_x).tolist()
y_lines = np.arange(y_min, y_max, grid_size_y).tolist()

x_lines.append(x_max)
y_lines.append(y_max)

for x in x_lines:
    fig.add_shape(type='line', x0=x ,y0=y_min, x1=x, y1=y_max,
                xref='x', yref='y',
                line=dict(color="Black", width=1))

for y in y_lines:
    fig.add_shape(type='line', x0=x_min ,y0=y, x1=x_max, y1=y,
                xref='x', yref='y',
                line=dict(color="Black", width=1))


###############################
# Initialize the array with zeros
for i in range(len(x_lines) - 1):  
    for j in range(len(y_lines) - 1): 
        label = f'{len(y_lines) - 2 -j}_{i}'
        study_df.loc[
            (study_df[x_label] >= x_lines[i]) & \
            (study_df[x_label] < x_lines[i+1]) & \
            (study_df[y_label] >= y_lines[j]) & \
            (study_df[y_label] < y_lines[j+1]), "grill"
        ] =  label

# 定義每個事件的權重
def weight(events):
    mapping = mapp
    return mapping.get(events, np.nan)  # 其他的都是nan

study_df["weighted_events"] = study_df["new_event"].apply(weight)

numerator = study_df.groupby('grill')['weighted_events']\
    .sum().dropna().to_dict()
denominator = study_df.groupby('grill').size().dropna().to_dict()

ratio = {}
for grill in denominator:
    if grill == 'nan':
        continue
    # if grill in numerator:
    #     ratio[grill] = numerator[grill]
    if grill in numerator:
        if denominator[grill] > 0:
            prob = numerator[grill] / denominator[grill]
            ratio[grill] = prob


A = [[0 for _ in range(len(x_lines) - 1)] for _ in range(len(y_lines) - 1)]
B = [[0 for _ in range(len(x_lines) - 1)] for _ in range(len(y_lines) - 1)]

for key in ratio:  
    parts = key.split('_')  
    if len(parts) == 2:  
        j, i = [int(part) for part in parts] 
        A[j][i] = ratio[key]

fig2 = px.imshow(A, color_continuous_scale=px.colors.sequential.RdBu_r,
                # zmax=1,
                # zmin=-1
                )
fig2.update_xaxes(tickvals=list(range(0, len(x_lines),5)),  
        ticktext=[x_lines[k] for k in range(0, len(x_lines), 5)],
        tickangle=15,  # = 0 keep the labels horizontal
        tickfont=dict(color='black'),
        ticks="outside", tickwidth=2, tickcolor="crimson",
        ticklen=10)
fig2.update_yaxes(tickvals=list(range(0, len(y_lines), 5)),  
        ticktext=[round(y_lines[k], 2) for k in 
                reversed(range(0, len(y_lines), 5))],
        tickangle=0,  # = 0 keep the labels horizontal
        tickfont=dict(color='black'),
        ticks="outside", tickwidth=2, tickcolor="crimson",
        ticklen=10)

fig2.update_layout(
    annotations=[
        dict(
            text=f"{extract_year(file_path)}",
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=12,
                color="black"
            ),
            align="center"
        )
    ],
    # coloraxis_colorbar=dict(
    #     tickvals=[-max_abs_value, 0, max_abs_value],
    #     ticktext=[str(-max_abs_value), '0', str(max_abs_value)]
    # )
)

fig2.write_html('heatmap.html', auto_open=True)
#%%
fig.update_layout(
    annotations=[
        dict(
            text=f"{extract_year(file_path)}",
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=12,
                color="black"
            ),
            align="center"
        )
    ]
)
fig.write_html('allpitcher.html', auto_open=True)

#%%