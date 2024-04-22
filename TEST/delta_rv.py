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


df22 = sort_data(df1)
df23 = sort_data(df2)

df22 = df22[["launch_angle", "launch_speed", "delta_run_exp"]]
df23 = df23[["launch_angle", "launch_speed", "delta_run_exp"]]

bins = [df["delta_run_exp"].min(),
        -0.2, 0, 0.5,
        df["delta_run_exp"].max()]



num_groups = len(bins) - 1
labels = [f"[{bins[i]:.3f}, {bins[i+1]:.3f})" for i in range(num_groups)]
color_map = {
    labels[0]: "red",
    labels[1]: 'blue',
    labels[2]: "green",
    labels[3]: "purple"
}
# cut資料並用categorical定義順序
df22["bin_group"] = pd.cut(df22["delta_run_exp"], bins=bins, labels=labels,
                        include_lowest=True)
df22["bin_group"] = pd.Categorical(df22["bin_group"], categories=labels,
                                ordered=True)

df23["bin_group"] = pd.cut(df23["delta_run_exp"], bins=bins, labels=labels,
                        include_lowest=True)
df23["bin_group"] = pd.Categorical(df23["bin_group"], categories=labels,
                                ordered=True)

df = pd.concat([df22, df23], ignore_index=True)

x_min, x_max = df[x_label].min(), df[x_label].max()
y_min, y_max = df[y_label].min(), df[y_label].max()

grid_size_x = (x_max - x_min) / x_grids
grid_size_y = (y_max - y_min) / y_grids
x_lines = np.arange(x_min, x_max, grid_size_x).tolist()
y_lines = np.arange(y_min, y_max, grid_size_y).tolist()

x_lines.append(x_max)
y_lines.append(y_max)



###########################################
# 製作互動式界面
app = Dash(__name__)

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
        df = df22  
        title = "Data for 2022"
    else:
        df = df23  
        title = "Data for 2023"

    fig = px.scatter(df, x="launch_angle", y="launch_speed",
                    color="bin_group",
                    color_discrete_map=color_map,
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
    app.run_server(debug=True, port=8090)
#%%
