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

file_path = r"C:\Users\user\Desktop\baseballdata\2022sppitchingdata.csv"
df = pd.read_csv(file_path)

#cole_2223 = cole_2223[cole_2223['pitch_type'] == 'FF']
#cole_2223 = cole_2223[cole_2223['year'] == year]
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

color_discrete_map = {
    'single': 'blue',
    'double': 'orange',
    'triple': 'green',
    'home_run': 'red',
    'field_out': 'purple',
    'others': 'grey'
}

fig = px.scatter(df, x=x_label, y=y_label, \
        color='new_events', \
        color_discrete_map=color_discrete_map,
symbol='new_events', 
marginal_x="histogram", marginal_y="histogram",
)
# fig.add_annotation(
#     x=max(cole_2223[x_label]),  # Position the annotation at the right-most point
#     y=max(cole_2223[y_label]),  # Position the annotation at the top-most point
#     text=f'Correlation: {corr:.2f}',  # Format the correlation to 2 decimal places
#     showarrow=False,
#     yshift=10,  # Shift the annotation slightly up for aesthetics
#     xanchor='right',  # Anchor the text to the right
#     bgcolor='white',  # Optional: provide a background color for the annotation text
#     bordercolor='black',  # Optional: provide a border color
#     borderwidth=1  # Optional: specify the border width
# )

bins_x = [df['launch_angle'].min(), -20, \
        0, 20, 40, df['launch_angle'].max()]
bins_y = [df['launch_speed'].min(), 70, \
        80, 100, df['launch_speed'].max()]


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
    # fig.add_shape(type="line", 
    #         x0=0, y0=bins_y[i], x1=1, y1=bins_y[i], 
    #         xref="paper", yref="y",  
    #         line=dict(color="RoyalBlue", width=1))
fig.write_html('scatter_plot.html', auto_open=True)

def update_scatter_plot(pitcher_id):
    # 根据选定的 pitcher ID 过滤 DataFrame
    filtered_df = df[df['pitcher'] == pitcher_id] if pitcher_id != 'All' else df

    # 创建散点图
    fig = px.scatter(filtered_df, x=x_label, y=y_label,
                color='new_events',
                symbol='new_events',
                marginal_x="histogram",
                marginal_y="histogram",
                color_discrete_map=color_discrete_map)
    
    # 将图表显示在 Notebook 中
    fig.show()


pitcher_ids = ['All'] + sorted(df['pitcher'].unique().tolist())
pitcher_dropdown = widgets.Dropdown(options=pitcher_ids, value='All', description='Pitcher ID:')


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