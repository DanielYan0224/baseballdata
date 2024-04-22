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
from dash import Dash, dcc, html, Input, Output, State 

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

def matrix_year(df, mapp):
    df['weighted_events'] = df['new_event'].apply(lambda event: mapp.get(event, np.nan))  # 使用新的 mapp 计算权重

    numerator = df.groupby('grill')['weighted_events'].sum().dropna().to_dict()
    denominator = df.groupby('grill').size().dropna().to_dict()

    ratio = {}
    for grill in denominator:
        if grill == 'nan':
            continue
        if grill in numerator and denominator[grill] > 0:
            prob = numerator[grill] / denominator[grill]
            ratio[grill] = prob

    A = [[0 for _ in range(len(x_lines) - 1)] for _ in range(len(y_lines) - 1)]
    for key in ratio:
        parts = key.split('_')
        if len(parts) == 2:
            j, i = [int(part) for part in parts]
            A[j][i] = ratio[key]
    return A

file_path_1 = r"C:\Users\user\Desktop\baseballdata\2022sppitchingdata.csv"
file_path_2 = r"C:\Users\user\Desktop\baseballdata\2023sppitchingdata.csv"

x_label = 'launch_angle'
y_label = 'launch_speed'

df1 = sort_data(load_dataframe(file_path_1))

df2 = sort_data(load_dataframe(file_path_2))

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
# 定義grid
x_min, x_max = df[x_label].min(), df[x_label].max()
y_min, y_max = df[y_label].min(), df[y_label].max()

grid_size_x = (x_max - x_min) / x_grids
grid_size_y = (y_max - y_min) / y_grids
x_lines = np.arange(x_min, x_max, grid_size_x).tolist()
y_lines = np.arange(y_min, y_max, grid_size_y).tolist()

x_lines.append(x_max)
y_lines.append(y_max)
#################################

df2022 = grill(sort_data(df1), x_lines, y_lines)
df2022["weighted_events"] = df2022["new_event"].apply(weight)
df2023 = grill(sort_data(df2), x_lines, y_lines)
df2023["weighted_events"] = df2023["new_event"].apply(weight)

#################################

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([ 
        html.Div([  
            html.Label(f"{key}:"),
            dcc.Input(id=f"{key}-input", type="number", 
                    value=mapp[key], style={'margin-right': '10px'})
        ], style={'display': 'flex', 'padding': '5px'}) for key in mapp.keys()
    ], style={'display': 'flex', 'flex-wrap': 'wrap',
            'justify-content': 'space-between'}),

    html.Button('Update Heatmap', id='update-button', n_clicks=0, 
                style={'width': '100%', 'margin-top': '20px'}),

    dcc.Dropdown(
        id='dataset-dropdown',
        options=[
            {'label': '2022 Data', 'value': '2022'},
            {'label': '2023 Data', 'value': '2023'}
        ],
        value='2022'
    ),

    dcc.Graph(id='heat-map')
])
@app.callback(
        
    Output('heat-map', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('dataset-dropdown', 'value')] +
    [State(f"{key}-input", 'value') for key in mapp.keys()]
)

def update_heatmap(n_clicks, selected_year, *mapp_values):
    # 更新mapp
    mapp_updated = dict(zip(mapp.keys(), mapp_values))

    # 年份選擇
    if selected_year == '2022':
        data = matrix_year(df2022, mapp_updated)  
        title = "Heatmap for 2022"
    else:
        data = matrix_year(df2023, mapp_updated)
        title = "Heatmap for 2023"

    fighp = px.imshow(data, 
            color_continuous_scale=px.colors.sequential.Blues)

    # 更新座標軸
    fighp.update_xaxes(
        tickvals=list(range(0, len(x_lines), 5)),
        ticktext=[x_lines[k] for k in range(0, len(x_lines), 5)],
        tickangle=15,
        tickfont=dict(color='black'),
        ticks="outside", tickwidth=2, tickcolor="crimson",
        ticklen=10
    )
    fighp.update_yaxes(
        tickvals=list(range(0, len(y_lines), 5)),
        ticktext=[round(y_lines[k], 2) for k in reversed(range(0, len(y_lines), 5))],
        tickangle=0,
        tickfont=dict(color='black'),
        ticks="outside", tickwidth=2, tickcolor="crimson",
        ticklen=10
    )

    fighp.update_layout(
        title={'text': title, 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}
    )

    return fighp

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
#%%