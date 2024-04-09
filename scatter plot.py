#%%
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from fractions import Fraction as F

#%%
import plotly.express as px
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

bins_x = [df[x_label].min(), \
        0, 40, df[x_label].max()]
bins_y = [df[y_label].min(), 60, 80, df[y_label].min()]

# 定義每個安打的價值
def new_event(events):
    mapping = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4, 'field_out': 0}
    return mapping.get(events, 0)  # 默认返回0

# 定義分割的區間
def assign_zone(df):
    if df['launch_angle'] < 0:
        if df['launch_speed'] < 60:
            return 'A'  # Zone A
        elif 60 <= df['launch_speed'] < 80:
            return 'B'  # Zone B
        else:
            return 'C'  # Zone C
    elif 0 <= df['launch_angle'] < 40:
        if 60 <= df['launch_speed'] < 80:
            return 'D'  # Zone D
    elif df['launch_angle'] >= 40:
        if 60 <= df['launch_speed'] < 80:
            return 'E'  # Zone E
    return 'Other'  # 其他区域

df['Zone'] = df.apply(assign_zone, axis=1)
df['event_numeric'] = df['events'].apply(new_event).astype(str)  # 确保这里的列名与您的DataFrame中的列名匹配

color_discrete_map = {
    '1': 'blue',   
    '2': 'orange',
    '3': 'green',
    '4': 'red',
    '0': 'purple'
}


fig = px.scatter(df, x=x_label, y=y_label, \
        color='event_numeric', \
        color_discrete_map=color_discrete_map,
        marginal_x="histogram", marginal_y="histogram",
)

fig.add_shape(type="line", 
        x0=df['launch_angle'].min(), y0=bins_y[1],
        x1=0, y1=bins_y[1], 
        xref="x", yref="y",  
        line=dict(color="black", width=1.5))

fig.add_shape(type="line", 
        x0=bins_x[1], y0=0, x1=bins_x[1], y1=1, 
        xref="x", yref="paper",  
        line=dict(color="black", width=1.5))

fig.add_shape(type="line", 
        x0=bins_x[2], y0=0, x1=bins_x[2], y1=1, 
        xref="x", yref="paper",  
        line=dict(color="black", width=1.5))

fig.add_shape(type="line", 
        x0=0, y0=bins_y[2], x1=bins_x[2], y1=bins_y[2], 
        xref="x", yref="y",  
        line=dict(color="black", width=1.5))

fig.write_html('allpitcher.html', auto_open=True)
#%%
widgets.interactive(update_scatter_plot, pitcher_id=pitcher_dropdown)
#%%
import plotly.graph_objects as go
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

# 加载数据和预处理...
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
    # 标记是否为选定的投手
    df['highlight'] = df['pitcher'] == pitcher_id
    
    # 使用 plotly.graph_objects 创建散点图，以便更细致地控制颜色
    fig = go.Figure()
    
    # 添加不是选定投手的数据点，颜色设置为灰色
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
    # 添加选定投手的数据点，使用 new_events 来设置颜色
    for event, group_df in df[df['highlight']].groupby('new_events'):
        fig.add_trace(go.Scatter(x=group_df[x_label], y=group_df[y_label], mode='markers',
        marker_symbol=symbol_map[event], 
        name=event, marker_color=color_discrete_map[event]))
    
    # 显示图表
    fig.show()

# 创建下拉菜单
pitcher_dropdown = widgets.Dropdown(
    options=[('Select a pitcher', None)] + [(str(id), id) for id in df['pitcher'].unique()],
    description='Pitcher ID:',
    value=None  # 默认没有选中任何投手
)

# 创建一个用于更新散点图的互动控件
interactive_plot = widgets.interactive(update_scatter_plot, pitcher_id=pitcher_dropdown)

# 显示下拉菜单
display(interactive_plot)

#%%