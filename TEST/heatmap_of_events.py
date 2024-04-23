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
from scipy.ndimage import median_filter

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

def scort_data (df):
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

df1 = scort_data(load_dataframe(file_path_1))

df2 = scort_data(load_dataframe(file_path_2))

df = pd.concat([df1, df2], axis=0)
df.head()
#################################
#可調整
file_path = file_path_1
study_df = load_dataframe(file_path)
x_grids = 40
y_grids = 40
mapp = {'single': 1, 
'double': 1, 
'triple': 1, 
'home_run': 1,
'DP': 0, 
'field_out': 0,
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



#################################
# Draw the scatter plot

# 定義grid
x_min, x_max = df[x_label].min(), df[x_label].max()
y_min, y_max = df[y_label].min(), df[y_label].max()

grid_size_x = (x_max - x_min) / x_grids
grid_size_y = (y_max - y_min) / y_grids
x_lines = np.arange(x_min, x_max, grid_size_x).tolist()
y_lines = np.arange(y_min, y_max, grid_size_y).tolist()

x_lines.append(x_max)
y_lines.append(y_max)

# for x in x_lines:
#     fig.add_shape(type='line', x0=x ,y0=y_min, x1=x, y1=y_max,
#                 xref='x', yref='y',
#                 line=dict(color="Black", width=1))

# for y in y_lines:
#     fig.add_shape(type='line', x0=x_min ,y0=y, x1=x_max, y1=y,
#                 xref='x', yref='y',
#                 line=dict(color="Black", width=1))




df2022 = grill(scort_data(df1), x_lines, y_lines)
df2022["weighted_events"] = df2022["new_event"].apply(weight)
df2023 = grill(scort_data(df2), x_lines, y_lines)
df2023["weighted_events"] = df2023["new_event"].apply(weight)


def matrix_year(df, mapp):
    numerator = df.groupby('grill')['weighted_events'].sum().dropna().to_dict()
    denominator = df.groupby('grill').size().dropna().to_dict()
    
    ratio = {}
    for grill in denominator:
        if grill == 'nan':
            continue
        if grill in numerator:
            if denominator[grill] > 0:
                prob = numerator[grill] / denominator[grill]
                ratio[grill] = prob
    
    A = [[0 for _ in range(len(x_lines) - 1)] for _ in range(len(y_lines) - 1)]
    for key in ratio:
        parts = key.split('_')
        if len(parts) == 2:
            j, i = [int(part) for part in parts]
            A[j][i] = ratio[key]

    return A

color_list = df2022['new_event'].map(color_discrete_map).tolist()
symbol_list = df2022['new_event'].map(symbol_map).tolist()
color_list = df2023['new_event'].map(color_discrete_map).tolist()
symbol_list = df2023['new_event'].map(symbol_map).tolist()



app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Div([
            # 顯示mapp的標籤
            html.Label(f"{key}:", style={'margin-right': '10px'}),  
            dcc.Input(
                # 輸入框的 ID
                id=f'input-{key}',
                # 輸入數字  
                type='number',
                # 默認為原本mapp的數字  
                value=value,  
                # 輸入框的外邊界
                style={'margin': '5px'} 
            )
            # 設置樣式 水平排列
        ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'})  
        for key, value in mapp.items() 
    ]),
    # 新增trigger
    html.Button('Update Heatmap', id='update-button', n_clicks=0), 
    dcc.Dropdown(
        # 下拉列表
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
    [State(f'input-{key}', 'value') for key in mapp.keys()]
)

def update_heatmap(n_clicks, selected_year, *values):
    updated_mapp = dict(zip(mapp.keys(), map(int, values)))

    if selected_year == '2022':
        df = df2022
    else:
        df = df2023


    df["new_event"] = df['events'].apply(new_event).astype(str) 
    df["weighted_events"] = df["new_event"]\
        .apply(lambda event: updated_mapp.get(event, np.nan))

    data = np.array(matrix_year(df, updated_mapp))
    title = f"Heatmap for {selected_year}"

    fighp = px.imshow(median_filter(data, size=5), 
                    color_continuous_scale=px.colors.sequential.Blues)
    fighp.update_layout(
        title={'text': title, 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}
    )

    for x in x_lines:
        fighp.add_shape(type='line', x0=x ,y0=y_min, x1=x, y1=y_max,
                xref='x', yref='y',
                line=dict(color="Black", width=1))

    for y in y_lines:
        fighp.add_shape(type='line', x0=x_min ,y0=y, x1=x_max, y1=y,
                xref='x', yref='y',
                line=dict(color="Black", width=1))

    return fighp

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
#%%