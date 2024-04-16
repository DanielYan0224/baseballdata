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