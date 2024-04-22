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
from plotly.graph_objs import Figure
import dash
from dash import Dash, dcc, html, Input, Output 

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

# 只留下安打與出局
def new_event(events):
    mapping = {'single': "single", 'double': "double", 
            'triple': "triple", 'home_run': "home_run", 
            'field_out': "field_out", "double_play": "DP"}
    return mapping.get(events, "others")  # 其他的都是others

def weight(events):
    mapping = mapp
    return mapping.get(events, np.nan)  # 其他的都是nan

def sort_data (df):
    df = df.dropna(subset=[x_label, y_label], how='any')
    df = df[df['description'] == 'hit_into_play']
    df['new_event'] = df['events'].apply(new_event).astype(str)
    return df

def grill(df, x_lines, y_lines):
    for i in range(len(x_lines) - 1):  
        for j in range(len(y_lines) - 1): 
            label = f'{len(y_lines) - 2 -j}_{i}'
            df.loc[
                (df[x_label] >= x_lines[i]) & \
                (df[x_label] < x_lines[i+1]) & \
                (df[y_label] >= y_lines[j]) & \
                (df[y_label] < y_lines[j+1]), "grill"
            ] =  label
    return df


file_path_1 = r"C:\Users\user\Desktop\baseballdata\2022sppitchingdata.csv"
file_path_2 = r"C:\Users\user\Desktop\baseballdata\2023sppitchingdata.csv"

x_label = 'launch_angle'
y_label = 'launch_speed'

df1 = sort_data(load_dataframe(file_path_1))

df2 = sort_data(load_dataframe(file_path_2))

df = pd.concat([df1, df2], axis=0)

# df1[[x_label, y_label, 'events']].sample(8)
# df2[[x_label, y_label, 'events']].sample(8)

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
x_min, x_max = df[x_label].min(), df[x_label].max()
y_min, y_max = df[y_label].min(), df[y_label].max()

grid_size_x = (x_max - x_min) / x_grids
grid_size_y = (y_max - y_min) / y_grids
x_lines = np.arange(x_min, x_max, grid_size_x).tolist()
y_lines = np.arange(y_min, y_max, grid_size_y).tolist()

x_lines.append(x_max)
y_lines.append(y_max)


app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='scatter-plot'),
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[
            {'label': '2022 Data', 'value': '2022'},
            {'label': '2023 Data', 'value': '2023'}
        ],
        value='2022'
    )
])

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('dataset-dropdown', 'value')]
)
def update_graph(selected_year):
    if selected_year == '2022':
        df = df1
        title = "Data for 2022"
    else:
        df = df2
        title = "Data for 2023"

    fig = px.scatter(df, x=x_label, y=y_label,
                    color='new_event', 
                    symbol='new_event',
                    color_discrete_map=color_discrete_map,
                    symbol_map=symbol_map,
                    marginal_x="histogram", 
                    marginal_y="histogram",
                    title=title)
    for x in x_lines:
        fig.add_shape(type='line', x0=x ,y0=y_min, x1=x, y1=y_max,
                xref='x', yref='y',
                line=dict(color="Black", width=1))

    for y in y_lines:
        fig.add_shape(type='line', x0=x_min ,y0=y, x1=x_max, y1=y,
                xref='x', yref='y',
                line=dict(color="Black", width=1))
        
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
#%%





#%%
import dash
from dash import html, dcc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd

data2022 = {
    'x': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'y': [9, 4, 2, 6, 8, 7, 5, 3, 1],
    'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A']
}
data2023 = {
    'x': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'y': [1, 3, 5, 7, 9, 8, 6, 4, 2],
    'category': ['B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
}

df2022 = pd.DataFrame(data2022)
df2023 = pd.DataFrame(data2023)

# 初始化 Dash 应用
app = dash.Dash(__name__)

# 应用布局
app.layout = html.Div([
    dcc.Graph(id='scatter-plot'),
    html.Button('Switch Data', id='switch-button', n_clicks=0)
])

# 应用回调
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('switch-button', 'n_clicks')]
)
def update_graph(n_clicks):
    if n_clicks % 2 == 0:
        df = df2022
        title = "Data for 2022"
    else:
        df = df2023
        title = "Data for 2023"

    fig = px.scatter(df, x='x', y='y', color='category', 
                    title=title)
    return fig

# 运行应用服务器
if __name__ == '__main__':
    app.run_server(debug=True)
#%%