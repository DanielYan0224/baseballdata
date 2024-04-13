#%%
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from fractions import Fraction as F

file_path = r"C:\Users\user\Desktop\baseballdata\2023sppitchingdata.csv"
df = pd.read_csv(file_path)

#cole_2223 = cole_2223[cole_2223['pitch_type'] == 'FF']
#cole_2223 = cole_2223[cole_2223['year'] == year]
df = df[df['description'] == 'hit_into_play']

x_label = 'launch_angle'
y_label = 'launch_speed'

# delete the data that has nan in the labels
df = df.dropna(subset=[x_label, y_label], how='any')

# 只留下安打與出局
def new_event(events):
    mapping = {'single': "single", 'double': "double", 
            'triple': "triple", 'home_run': "home_run", 
            'field_out': "field_out"}
    return mapping.get(events, "others")  # 其他的都是0
df['new_event'] = df['events'].apply(new_event).astype(str)  # 确保这里的列名与您的DataFrame中的列名匹配

symbol_map = {
    'single': 'circle',
    'double': 'square',
    'triple': 'diamond',
    'home_run': 'star',
    'field_out': 'x',
    'others': 'triangle-up'
}
color_discrete_map = {
    'single': 'blue',
    'double': 'orange',
    'triple': 'green',
    'home_run': 'red',
    'field_out': 'purple',
    'others': 'grey'
}

# 新增標線
fig = px.scatter(df, x=x_label, y=y_label, \
        color='new_event', \
        color_discrete_map=color_discrete_map,
        marginal_x="histogram", marginal_y="histogram",
)

# 定義grid size
grid_size = 5.0
x_min, x_max = df[x_label].min(), df[x_label].max()
y_min, y_max = df[y_label].min(), df[y_label].max()

x_lines = np.arange(x_min, x_max + grid_size, grid_size).tolist()

y_lines = np.arange(y_min, y_max + grid_size, grid_size).tolist()

# check x_max and y_max
if x_max not in x_lines:
    x_lines.append(x_max)
if y_max not in y_lines:
    y_lines.append(y_max)

for x in x_lines:
    fig.add_shape(type='line', x0=x ,y0=y_min, x1=x, y1=y_max,
                xref='x', yref='y',
                line=dict(color="Black", width=1))

for y in y_lines:
    fig.add_shape(type='line', x0=x_min ,y0=y, x1=x_max, y1=y,
                xref='x', yref='y',
                line=dict(color="Black", width=1))

#fig.write_html('allpitcher.html', auto_open=True)

###############################
# Initialize the array with zeros
for i in range(len(x_lines) - 1):  
    for j in range(len(y_lines) - 1): 
        label = f'{len(y_lines) - 2 -j}_{i}'
        df.loc[
            (df[x_label] >= x_lines[i]) & \
            (df[x_label] < x_lines[i+1]) & \
            (df[y_label] >= y_lines[j]) & \
            (df[y_label] < y_lines[j+1]), "grill"
        ] =  label

# 定義每個事件的權重
def weight(events):
    mapping = {'single': 1, 'double': 2, 
            'triple': 3, 'home_run': 4, 
            'field_out': 0}
    return mapping.get(events, 0)  # 其他的都是0

df["weighted_events"] = df["events"].apply(weight)

grill_weighted_events = df.groupby('grill')['weighted_events'].sum().dropna()

A = [[0 for _ in range(len(x_lines) - 1)] for _ in range(len(y_lines) - 1)]

for grill, weight in grill_weighted_events.items():
    if grill == 'nan':
        continue  
    j, i = map(int, grill.split('_')) 
    A[j][i] = weight

print(grill_weighted_events)
for row in A:
    print(row)
#%%
import plotly.graph_objects as go
import pandas as pd
import ipywidgets as widgets
from IPython.display import display


file_path = r"C:\Users\user\Desktop\baseballdata\2022sppitchingdata.csv"
df = pd.read_csv(file_path)
df = df[df['description'] == 'hit_into_play']

x_label = 'launch_angle'
y_label = 'launch_speed'

def new_event(events):
    if events == 'single':
        return 'single'
    elif events == 'double':
        return 'double'
    elif events == 'triple':
        return 'triple'
    elif events == 'home_run':
        return 'home_run'
    elif events == 'field_out':
        return 'field_out'
    else:
        return 'others'
df['new_events'] = df['events'].apply(new_event)

def update_scatter_plot(pitcher_id):
    # 標記選定的選手
    df['highlight'] = df['pitcher'] == pitcher_id
    
    # 
    fig = go.Figure()
    
    # 非選定選手的資料
    fig.add_trace(go.Scatter(x=df[~df['highlight']][x_label], y=df[~df['highlight']][y_label],
                        mode='markers', marker_color='lightgrey', name='Others'))
    symbol_map = {
    'single': 'circle',
    'double': 'square',
    'triple': 'diamond',
    'home_run': 'star',
    'field_out': 'x',
    'others': 'triangle-up'
}
    color_discrete_map = {
    'single': 'blue',
    'double': 'orange',
    'triple': 'green',
    'home_run': 'red',
    'field_out': 'purple',
    'others': 'grey'
}

    for event, group_df in df[df['highlight']].groupby('new_events'):
        fig.add_trace(go.Scatter(x=group_df[x_label], y=group_df[y_label], mode='markers',
        marker_symbol=symbol_map[event], 
        name=event, marker_color=color_discrete_map[event]))
    fig.show()

pitcher_dropdown = widgets.Dropdown(
    options=[('Select a pitcher', None)] + [(str(id), id) for id in df['pitcher'].unique()],
    description='Pitcher ID:',
    value=None 
)

interactive_plot = widgets.interactive(update_scatter_plot, pitcher_id=pitcher_dropdown)

display(interactive_plot)

#%%