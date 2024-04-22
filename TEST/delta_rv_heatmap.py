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

def sort_data (df):
    df = df.dropna(subset=[x_label, y_label], how='any')
    df = df[df['description'] == 'hit_into_play']
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

def clean_id(raw_id):
    # 為id清除小數點
    return raw_id.replace('.', '_').replace('[', '').replace(']', '')\
    .replace(',', '_').replace('-', '_')

def matrix_rv(df, labels, updated_mapp):
    df["interval_rv"] = pd.cut(df["delta_run_exp"], bins=bins, labels=labels, include_lowest=True)
    df["grouped_rv"] = df["interval_rv"].map(updated_mapp)  # 使用更新后的映射

    group_counts = df.groupby('grill').size().to_dict()
    total_counts = df.groupby('grill').size().to_dict()

    ratio = {}
    for grill in total_counts:
        if grill == 'nan':
            continue
        if grill in group_counts and total_counts[grill] > 0:
            prob = group_counts[grill] / total_counts[grill]
            ratio[grill] = prob

    B = [[0 for _ in range(len(x_lines) - 1)] for _ in range(len(y_lines) - 1)]
    for key in ratio:
        parts = key.split('_')
        if len(parts) == 2:
            j, i = [int(part) for part in parts]
            B[j][i] = ratio[key]

    return B



###########################################
file_path_1 = r"C:\Users\user\Desktop\baseballdata\2022sppitchingdata.csv"
file_path_2 = r"C:\Users\user\Desktop\baseballdata\2023sppitchingdata.csv"

###########################################
x_label = "launch_angle"
y_label = "launch_speed"
x_grids = 40
y_grids = 40

###########################################

df1 = sort_data(load_dataframe(file_path_1))

df2 = sort_data(load_dataframe(file_path_2))

df = pd.concat([df1, df2], axis=0)

df22 = df1[["launch_angle", "launch_speed", "delta_run_exp"]]
df23 = df2[["launch_angle", "launch_speed", "delta_run_exp"]]

df = pd.concat([df22, df23], axis=0)
###########################################
# 定義grid
x_min, x_max = df[x_label].min(), df[x_label].max()
y_min, y_max = df[y_label].min(), df[y_label].max()

grid_size_x = (x_max - x_min) / x_grids
grid_size_y = (y_max - y_min) / y_grids
x_lines = np.arange(x_min, x_max, grid_size_x).tolist()
y_lines = np.arange(y_min, y_max, grid_size_y).tolist()

x_lines.append(x_max)
y_lines.append(y_max)


###########################################
bins = [df["delta_run_exp"].min(),
        -0.2, 0, 0.5,
        df["delta_run_exp"].max()]

num_groups = len(bins) - 1
labels = [f"[{bins[i]:.3f}, {bins[i+1]:.3f})" 
        for i in range(num_groups)]
group = {labels[i]: i+1 for i in range(num_groups)}
mapp = {
    labels[0]: 1,
    labels[1]: 1,
    labels[2]: 1,
    labels[3]: 1
}

###########################################
df22 = grill(df22, x_lines, y_lines)
df23 = grill(df23, x_lines, y_lines)
###########################################
input_components = [
    html.Div([
        html.Label(f"{key}:"),
        dcc.Input(
            id=clean_id(f"{key}-input"),
            type="number",
            value=mapp[key],
            style={'margin-right': '10px'}
        )
    ], style={'display': 'flex', 'padding': '5px'})
    for key in mapp.keys()
]

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div(input_components, style={'display': 'flex', 'flex-wrap': 'wrap', 
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
    [State(clean_id(f"{key}-input"), 'value') for key in mapp.keys()]
)
def update_heatmap(n_clicks, selected_year, *mapp_values):
    updated_mapp = {labels[i]: value for i, value in enumerate(mapp_values)}
    
    if selected_year == '2022':
        df = df22
    else:
        df = df23

    data = matrix_rv(df, labels, updated_mapp)  # 确保传入 updated_mapp
    title = f"Heatmap for {selected_year}"

    fighp = px.imshow(data, color_continuous_scale=px.colors.sequential.Blues)
    fighp.update_layout(
        title={'text': title, 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}
    )

    return fighp

if __name__ == '__main__':
    app.run_server(debug=True)
#%%