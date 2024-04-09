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

# 只留下安打與出局
def new_event(events):
    mapping = {'single': "single", 'double': "double", 
            'triple': "triple", 'home_run': "home_run", 
            'field_out': "field_out"}
    return mapping.get(events, "others")  # 其他的都是0
df['new_event'] = df['events'].apply(new_event).astype(str)  # 确保这里的列名与您的DataFrame中的列名匹配


# 新增標線
fig = px.scatter(df, x=x_label, y=y_label, \
        color='new_event', \
        color_discrete_map=color_discrete_map,
        marginal_x="histogram", marginal_y="histogram",
)

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

# Classify zone by LA and LS
df.loc[(df[x_label] < 0) & (df[y_label] < 60), 'classify_zone'] = "A"
df.loc[(df[x_label] < 0) & (df[y_label] >= 60), 'classify_zone'] = "B"
df.loc[(0 <= df[x_label]) & (df[x_label] < 40) &
    (bins_y[0] <= df[y_label]) & (df[y_label] <= 80)
    , "classify_zone"] = "C"
df.loc[(0 <= df[x_label]) & (df[x_label] < 40) \
    & (80 < df[y_label] ) & (df[y_label] <= bins_y[3])
    , 'classify_zone'] = 'D'
df.loc[(40 <= df[x_label]), 'classify_zone'] = "E"


df_E = df[df["classify_zone"] == "E"]
print(df_E["new_event"].value_counts(normalize= True))
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